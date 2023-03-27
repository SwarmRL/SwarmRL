"""
Run an RL agent to find the center of a box.
"""
import copy
import tempfile
import unittest as ut

import flax.linen as nn
import numpy as np
import optax
import pint

import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.models.interaction_model import Action
from swarmrl.utils import utils


class TestRLScript(ut.TestCase):
    simulation_name = "example_simulation"

    def rl_simulation(self, outfolder):
        """
        Run the simulation.

        Returns
        -------

        """
        loglevel_terminal = "info"
        seed = 42

        # manually turn on or off, cannot be checked in a test case
        logger = utils.setup_swarmrl_logger(
            f"{outfolder}/{self.simulation_name}.log",
            loglevel_terminal=loglevel_terminal,
        )
        logger.info("Starting simulation setup")

        ureg = pint.UnitRegistry()
        md_params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(297.0, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(300.0, "kelvin"),
            box_length=ureg.Quantity(1000, "micrometer"),
            time_slice=ureg.Quantity(0.5, "second"),  # model timestep
            time_step=ureg.Quantity(0.5, "second") / 5,  # integrator timestep
            write_interval=ureg.Quantity(2, "second"),
        )

        # parameters needed for bechinger_models.Baeuerle2020
        model_params = {
            "target_vel_SI": ureg.Quantity(0.5, "micrometer / second"),
            "target_ang_vel_SI": ureg.Quantity(4 * np.pi / 180, "1/second"),
            "vision_half_angle": np.pi,
            "detection_radius_position_SI": ureg.Quantity(np.inf, "meter"),
            "detection_radius_orientation_SI": ureg.Quantity(25, "micrometer"),
            "angular_deviation": 67.5 * np.pi / 180,
        }

        run_params = {
            "n_colloids": 10,
            "sim_duration": ureg.Quantity(3, "minute"),
            "seed": seed,
        }

        # from now on, no new parameters are introduced

        system_runner = srl.espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=run_params["seed"],
            out_folder=outfolder,
            write_chunk_size=100,
        )

        coll_type = 0
        system_runner.add_colloids(
            run_params["n_colloids"],
            ureg.Quantity(2.14, "micrometer"),
            ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
            ureg.Quantity(400, "micrometer"),
            type_colloid=coll_type,
        )

        md_params_without_ureg = copy.deepcopy(md_params)
        md_params_without_ureg.ureg = None

        # Define the force model.

        class ActorNet(nn.Module):
            """A simple dense model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=128)(x)
                x = nn.relu(x)
                x = nn.Dense(features=4)(x)
                return x

        class CriticNet(nn.Module):
            """A simple dense model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=128)(x)
                x = nn.relu(x)
                x = nn.Dense(features=1)(x)
                return x

        # Define networks
        critic_stack = CriticNet()
        actor_stack = ActorNet()

        # Define an exploration policy
        exploration_policy = srl.exploration_policies.RandomExploration(probability=0.1)

        # Define a sampling_strategy
        sampling_strategy = srl.sampling_strategies.GumbelDistribution()

        # Value function to use
        value_function = srl.value_functions.ExpectedReturns(
            gamma=0.99, standardize=True
        )

        # Define the models.
        actor = srl.networks.FlaxModel(
            flax_model=actor_stack,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
            sampling_strategy=sampling_strategy,
            exploration_policy=exploration_policy,
        )
        critic = srl.networks.FlaxModel(
            flax_model=critic_stack,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
        )

        def scale_function(distance: float):
            """
            Scaling function for the task
            """
            return 1 - distance

        task = srl.tasks.searching.GradientSensing(
            source=np.array([500.0, 500.0, 0.0]),
            decay_function=scale_function,
            reward_scale_factor=10,
            box_length=np.array([1000.0, 1000.0, 1000]),
        )
        task.initialize(system_runner.colloids)

        observable = srl.observables.ConcentrationField(
            source=np.array([500.0, 500.0, 0.0]),
            decay_fn=scale_function,
            scale_factor=10000,
            box_length=np.array([1000.0, 1000.0, 1000]),
            particle_type=0,
        )
        observable.initialize(system_runner.colloids)

        # Define the loss model
        loss = srl.losses.PolicyGradientLoss(value_function=value_function)
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
        protocol = srl.rl_protocols.ActorCritic(
            particle_type=0,
            actor=actor,
            critic=critic,
            task=task,
            observable=observable,
            actions=actions,
        )
        # Define the force model.
        rl_trainer = srl.gyms.Gym(
            [protocol],
            loss,
        )

        params_to_write = {
            "md_params": md_params_without_ureg,
            "model_params": model_params,
            "run_params": run_params,
        }

        utils.write_params(
            outfolder,
            self.simulation_name,
            params_to_write,
            write_espresso_version=True,
        )

        # Run the simulation.
        logger.info("starting simulation")
        n_slices = int(run_params["sim_duration"] / md_params.time_slice)
        n_episodes = 2
        episode_length = int(np.ceil(n_slices / 60))  # 15 steps
        rl_trainer.perform_rl_training(
            system_runner=system_runner,
            n_episodes=n_episodes,
            episode_length=episode_length,
            initialize=True,
        )

    def test_full_sim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.rl_simulation(outfolder)


if __name__ == "__main__":
    ut.main()
