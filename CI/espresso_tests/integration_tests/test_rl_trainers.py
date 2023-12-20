import glob
import tempfile
import unittest as ut

import espressomd
import flax.linen as nn
import numpy as np
import optax
import pint

import swarmrl as srl
from swarmrl.actions import Action
from swarmrl.engine import espresso


class ActoCriticNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        y = nn.Dense(features=1)(x)
        x = nn.Dense(features=4)(x)
        return x, y


def scale_function(distance: float):
    """
    Scaling function for the task
    """
    return 1 - distance


translate = Action(force=10.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action()

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}


class KillTask(srl.tasks.Task):
    """
    Dummy task for the tests.

    This task will turn on the killswitch after
    4 episodes.
    """

    i: int = 0

    def __call__(self, colloids) -> float:
        if self.i == 15:
            self.kill_switch = True
        self.i += 1

        return np.random.uniform(size=(5,))


class EspressoTestRLTrainers(ut.TestCase):
    """
    Tests all of the SwarmRL trainers.
    """

    system = espressomd.System(box_l=[1, 2, 3])
    seed = 42

    # MD parameters
    ureg = pint.UnitRegistry()
    md_params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
        WCA_epsilon=ureg.Quantity(297.0, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(300.0, "kelvin"),
        box_length=ureg.Quantity(1000, "micrometer"),
        time_slice=ureg.Quantity(10.0, "second"),  # model timestep
        time_step=ureg.Quantity(0.1, "second"),  # integrator timestep
        write_interval=ureg.Quantity(1.0, "second"),
    )

    # Define networks
    actor_critic = ActoCriticNet()

    # Define an exploration policy
    exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.GumbelDistribution()

    # Define the models.
    network = srl.networks.FlaxModel(
        flax_model=actor_critic,
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(1,),
        sampling_strategy=sampling_strategy,
        exploration_policy=exploration_policy,
    )
    task = srl.tasks.searching.GradientSensing(
        source=np.array([500.0, 500.0, 0.0]),
        decay_function=scale_function,
        reward_scale_factor=10,
        box_length=np.array([1000.0, 1000.0, 1000]),
    )

    observable = srl.observables.ConcentrationField(
        source=np.array([500.0, 500.0, 0.0]),
        decay_fn=scale_function,
        scale_factor=10000,
        box_length=np.array([1000.0, 1000.0, 1000]),
        particle_type=0,
    )

    # Define the loss model
    loss = srl.losses.ProximalPolicyLoss()
    agent = srl.agents.ActorCriticAgent(
        particle_type=0,
        network=network,
        task=task,
        observable=observable,
        actions=actions,
    )

    def test_continuous_training(self):
        """
        Test continuous training.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            system_runner = srl.espresso.EspressoMD(
                md_params=self.md_params,
                n_dims=2,
                seed=self.seed,
                out_folder=f"{temp_dir}/{self.seed}",
                system=self.system,
                write_chunk_size=10,
            )

            system_runner.add_colloids(
                5,
                self.ureg.Quantity(2.14, "micrometer"),
                self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                self.ureg.Quantity(400, "micrometer"),
                type_colloid=0,
            )

            # Define the force model.
            rl_trainer = srl.trainers.ContinuousTrainer(
                [self.agent],
            )
            rl_trainer.perform_rl_training(
                system_runner=system_runner,
                n_episodes=5,
                episode_length=5,
            )

    def test_continous_training_early_stop(self):
        """
        Test the case where the task ends the training.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            system_runner = srl.espresso.EspressoMD(
                md_params=self.md_params,
                n_dims=2,
                seed=self.seed,
                out_folder=f"{temp_dir}/{self.seed}",
                system=self.system,
                write_chunk_size=1,
            )

            system_runner.add_colloids(
                5,
                self.ureg.Quantity(2.14, "micrometer"),
                self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                self.ureg.Quantity(400, "micrometer"),
                type_colloid=0,
            )
            # We need a custom protoc0l for this test.
            agent = srl.agents.ActorCriticAgent(
                particle_type=0,
                network=self.network,
                task=KillTask(),
                observable=self.observable,
                actions=actions,
            )

            # Define the force model.
            rl_trainer = srl.trainers.ContinuousTrainer(
                [agent],
                self.loss,
            )
            rewards = rl_trainer.perform_rl_training(
                system_runner=system_runner,
                n_episodes=100,  # Should stop after 4
                episode_length=5,
            )
            assert rewards.shape == (4,)  # Should only have four steps.

    def test_fixed_episodic_training(self):
        """
        Test the episodic training for set episode length.

        The training should dump 10 trajectories.
        """
        with tempfile.TemporaryDirectory() as temp_dir:

            def get_engine(system):
                """
                Get the engine.
                """
                seed = np.random.randint(89675392)
                system_runner = srl.espresso.EspressoMD(
                    md_params=self.md_params,
                    n_dims=2,
                    seed=seed,
                    out_folder=f"{temp_dir}/episodic/{seed}",
                    system=self.system,
                    write_chunk_size=10,
                )

                coll_type = 0
                system_runner.add_colloids(
                    5,
                    self.ureg.Quantity(2.14, "micrometer"),
                    self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                    self.ureg.Quantity(400, "micrometer"),
                    type_colloid=coll_type,
                )

                return system_runner

            # Define the force model.
            rl_trainer = srl.trainers.EpisodicTrainer(
                [self.agent],
            )

            rl_trainer.perform_rl_training(
                get_engine=get_engine,
                n_episodes=10,
                system=self.system,
                episode_length=10,
            )

            dumped_files = glob.glob(f"{temp_dir}/episodic/*")
            assert len(dumped_files) == 10

    def test_variable_episodic_training(self):
        """
        Test episodic training with engine killing tasks.
        """
        with tempfile.TemporaryDirectory() as temp_dir:

            def get_engine(system):
                """
                Get the engine.
                """
                seed = np.random.randint(89675392)
                system_runner = srl.espresso.EspressoMD(
                    md_params=self.md_params,
                    n_dims=2,
                    seed=seed,
                    out_folder=f"{temp_dir}/episodic/{seed}",
                    system=self.system,
                    write_chunk_size=10,
                )

                coll_type = 0
                system_runner.add_colloids(
                    5,
                    self.ureg.Quantity(2.14, "micrometer"),
                    self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                    self.ureg.Quantity(400, "micrometer"),
                    type_colloid=coll_type,
                )

                return system_runner

            # We need a custom protoc0l for this test.
            agent = srl.agents.ActorCriticAgent(
                particle_type=0,
                network=self.network,
                task=KillTask(),
                observable=self.observable,
                actions=actions,
            )

            # Define the force model.
            rl_trainer = srl.trainers.EpisodicTrainer(
                [agent],
            )

            rl_trainer.perform_rl_training(
                get_engine=get_engine,
                n_episodes=10,
                reset_frequency=1000,  # Will only be reset after a failure.
                system=self.system,
                episode_length=10,
            )

            dumped_files = glob.glob(f"{temp_dir}/episodic/*")
            assert len(dumped_files) == 2

    def test_semi_episodic_training(self):
        """
        Test semi-episodic training.

        The system should be reset 5 times during the training.
        """
        with tempfile.TemporaryDirectory() as temp_dir:

            def get_engine(system):
                """
                Get the engine.
                """
                seed = np.random.randint(89675392)
                system_runner = srl.espresso.EspressoMD(
                    md_params=self.md_params,
                    n_dims=2,
                    seed=seed,
                    out_folder=f"{temp_dir}/episodic/{seed}",
                    system=self.system,
                    write_chunk_size=1,
                )

                coll_type = 0
                system_runner.add_colloids(
                    5,
                    self.ureg.Quantity(2.14, "micrometer"),
                    self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                    self.ureg.Quantity(400, "micrometer"),
                    type_colloid=coll_type,
                )

                return system_runner

            # Define the force model.
            rl_trainer = srl.trainers.EpisodicTrainer(
                [self.agent],
                self.loss,
            )
            rl_trainer.perform_rl_training(
                get_engine=get_engine,
                n_episodes=10,
                system=self.system,
                reset_frequency=2,
                episode_length=10,
            )

            dumped_files = glob.glob(f"{temp_dir}/episodic/*")
            assert len(dumped_files) == 5


if __name__ == "__main__":
    ut.main()
