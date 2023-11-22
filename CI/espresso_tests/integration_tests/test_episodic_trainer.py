"""
Run an RL agent to find the center of a box.
"""

# import copy
# import tempfile
# import unittest as ut

import espressomd
import flax.linen as nn
import numpy as np
import optax
import pint

import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.models.interaction_model import Action

# from swarmrl.utils import utils

# class TestRLScript(ut.TestCase):
#     simulation_name = "example_simulation"


def rl_simulation(outfolder):
    """
    Run the simulation.

    Returns
    -------

    """
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

    # from now on, no new parameters are introduced
    def get_engine(system):
        """
        Get the engine.
        """
        seed = np.random.randint(89675392)
        system_runner = srl.espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=seed,
            out_folder=f"episodic/{seed}",
            system=system,
            write_chunk_size=10,
        )

        coll_type = 0
        system_runner.add_colloids(
            50,
            ureg.Quantity(2.14, "micrometer"),
            ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
            ureg.Quantity(400, "micrometer"),
            type_colloid=coll_type,
        )

        return system_runner

    # Define the force model.

    class ActoCriticNet(nn.Module):
        """A simple dense model."""

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=12)(x)
            x = nn.relu(x)
            y = nn.Dense(features=1)(x)
            x = nn.Dense(features=4)(x)
            return x, y

    # Define networks
    acto_critic = ActoCriticNet()

    # Define an exploration policy
    exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.GumbelDistribution()

    # Define the models.
    network = srl.networks.FlaxModel(
        flax_model=acto_critic,
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(1,),
        sampling_strategy=sampling_strategy,
        exploration_policy=exploration_policy,
    )

    def scale_function(distance: float):
        """
        Scaling function for the task
        """
        return 1 - distance

    task = srl.tasks.searching.GradientSensing(
        source=np.array([500.0, 500.0, 0.0]),
        decay_function=scale_function,
        reward_scale_factor=1000,
        box_length=np.array([1000.0, 1000.0, 1000]),
    )

    observable = srl.observables.ConcentrationField(
        source=np.array([500.0, 500.0, 0.0]),
        decay_fn=scale_function,
        scale_factor=1000,
        box_length=np.array([1000.0, 1000.0, 1000]),
        particle_type=0,
    )

    # Define the loss model
    loss = srl.losses.ProximalPolicyLoss()
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
        network=network,
        task=task,
        observable=observable,
        actions=actions,
    )
    # Define the force model.
    rl_trainer = srl.gyms.EpisodicTrainer(
        [protocol],
        loss,
    )

    # Run the simulation.
    n_episodes = 500
    episode_length = 60
    system = espressomd.System(box_l=[1000, 1000, 1000])
    rl_trainer.perform_rl_training(
        get_engine=get_engine,
        n_episodes=n_episodes,
        system=system,
        episode_length=episode_length,
    )

    # def test_full_sim(self):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
    #         self.rl_simulation(outfolder)


if __name__ == "__main__":
    rl_simulation(".")
#     ut.main()
