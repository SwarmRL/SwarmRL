"""
Run an RL agent to find the center of a box.
"""
import copy
import unittest as ut

import flax.linen as nn
import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
import optax
import pint

import swarmrl as srl
import swarmrl.engine.espresso as espresso


class TestFindCenter(ut.TestCase):
    """
    Functional test, also to be used as an example for simulation scripts
    """

    def run_analysis(folder_name: str):
        """
        Run some analysis.

        Returns
        -------

        """
        with hf.File(f"example_output/{folder_name}/trajectory.hdf5") as db:
            data = np.array(db["colloids"]["Unwrapped_Positions"])
        for i in range(len(data[0])):
            plt.plot(data[:, i][:, 0], data[:, i][:, 1])

        plt.show()

    def visualize_particles(folder_name: str):
        """
        Run a visualization of the particles in the database.

        Returns
        -------

        """
        import znvis as vis

        with hf.File(f"example_output/{folder_name}/trajectory.hdf5") as db:
            data = np.array(db["colloids"]["Unwrapped_Positions"])

        mesh = vis.Sphere(
            radius=10.0, colour=np.array([30, 144, 255]) / 255, resolution=5
        )
        colloids = vis.Particle(name="Colloid", mesh=mesh, position=data)

        visualizer = vis.Visualizer(particles=[colloids], frame_rate=40)
        visualizer.run_visualization()

    def run_simulation(
        self, exploration_prob, cone_reward_scale_factor, vision_angle, folder_name
    ):
        """
        Run the simulation.

        Returns
        -------

        """
        loglevel_terminal = "info"
        seed = 42

        outfolder = srl.utils.setup_sim_folder(
            "./example_output", folder_name, ask_if_exists=not "store_true"
        )
        logger = srl.utils.setup_swarmrl_logger(
            f"{outfolder}/{folder_name}.log", loglevel_terminal=loglevel_terminal
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
        exploration_policy = srl.exploration_policies.RandomExploration(
            probability=exploration_prob
        )

        # Define a sampling_strategy
        sampling_strategy = srl.sampling_strategies.GumbelDistribution()

        # Value function to use
        value_function = srl.value_functions.ExpectedReturns(
            gamma=0.99, standardize=True
        )

        # Define the models.
        actor = srl.networks.FlaxModel(
            flax_model=actor_stack,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(2,),
            sampling_strategy=sampling_strategy,
            exploration_policy=exploration_policy,
        )
        critic = srl.networks.FlaxModel(
            flax_model=critic_stack,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(2,),
        )

        def scale_function(distance: float):
            """
            Scaling function for the task
            """
            return 1 - distance

        task = srl.tasks.searching.GradientSensingVisionCone(
            source=np.array([500.0, 500.0, 0.0]),
            decay_function=scale_function,
            grad_reward_scale_factor=10,
            box_size=np.array([1000.0, 1000.0, 1000]),
            cone_reward_scale_factor=cone_reward_scale_factor,
            vision_angle=vision_angle,
            vision_direction=complex(0, 1),
        )

        observable = task.init_task()

        # Define the loss model
        loss = srl.losses.PolicyGradientLoss(value_function=value_function)

        # Define the force model.
        rl_trainer = srl.gyms.Gym(
            actor,
            critic,
            task,
            loss,
            observable,
            run_params["n_colloids"],
        )

        # Run the simulation.
        logger.info("starting simulation")
        n_slices = int(run_params["sim_duration"] / md_params.time_slice)

        n_episodes = 2
        episode_length = int(np.ceil(n_slices / 900))

        rl_trainer.perform_rl_training(
            system_runner=system_runner,
            n_episodes=n_episodes,
            episode_length=episode_length,
            initialize=True,
        )

    def test_find_center(self):
        exploration_prob = 0.2
        cone_reward_scale_factor, vision_angle = 0.2, 30
        # folder_name = 'scale' + str(int(cone_reward_scale_factor * 1000)) + \
        #               'angle' + str(vision_angle)
        folder_name = "test"
        self.run_simulation(
            exploration_prob, cone_reward_scale_factor, vision_angle, folder_name
        )

        with hf.File(f"./example_output/{folder_name}/trajectory.hdf5") as db:
            data = np.array(db["colloids"]["Unwrapped_Positions"])
        assert len(data[0]) > 0


if __name__ == "__main__":
    ut.main()
