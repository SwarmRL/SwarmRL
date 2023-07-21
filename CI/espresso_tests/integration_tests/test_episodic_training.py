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


class TestRLEpisodicScript(ut.TestCase):
    simulation_name = "example_simulation"
    with tempfile.TemporaryDirectory() as temp_dir:
        outfolder = utils.setup_sim_folder(temp_dir, simulation_name)

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
        "seed": 42,
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

    def rl_simulation(self, outfolder):
        """
        Run the simulation.

        Returns
        -------

        """
        md_params_without_ureg = copy.deepcopy(self.md_params)
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
        task.initialize(self.system_runner.colloids)

        observable = srl.observables.ConcentrationField(
            source=np.array([500.0, 500.0, 0.0]),
            decay_fn=scale_function,
            scale_factor=10000,
            box_length=np.array([1000.0, 1000.0, 1000]),
            particle_type=0,
        )
        observable.initialize(self.system_runner.colloids)

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
            "model_params": self.model_params,
            "run_params": self.run_params,
        }

        utils.write_params(
            outfolder,
            self.simulation_name,
            params_to_write,
            write_espresso_version=True,
        )

        # Run the simulation.
        n_slices = int(self.run_params["sim_duration"] / self.md_params.time_slice)
        n_episodes = 10
        episode_length = int(np.ceil(n_slices / 60))  # 15 steps
        rl_trainer.perform_rl_training(
            system_runner=self.system_runner,
            n_episodes=n_episodes,
            episode_length=episode_length,
            episodic_training=True,
        )

    def test_reset(self):
        # test if the system is reset and the number of colloids is the same
        num_of_cols_old = len(self.system_runner.colloids)
        self.system_runner.reset_system()
        num_of_cols_new = len(self.system_runner.colloids)
        self.assertEqual(num_of_cols_old, num_of_cols_new)

    def test_add_colloids(self):
        # add a different type of colloids and check if the number of colloids is
        # correct
        # also check if the ids and the types are correct

        coll_type = 1
        self.system_runner.add_colloids(
            10,
            self.ureg.Quantity(2.14, "micrometer"),
            self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
            self.ureg.Quantity(400, "micrometer"),
            type_colloid=coll_type,
        )

        # check if it is the correct number of colloids
        self.assertEqual(len(self.system_runner.colloids), 20)

        colloids_old = copy.deepcopy(self.system_runner.colloids)
        ids_old = [colloid.id for colloid in colloids_old]
        types_old = [colloid.type for colloid in colloids_old]

        self.system_runner.reset_system()

        colloids_new = copy.deepcopy(self.system_runner.colloids)
        ids_new = [colloid.id for colloid in colloids_new]
        types_new = [colloid.type for colloid in colloids_new]
        self.assertEqual(len(self.system_runner.colloids), 20)
        self.assertEqual(ids_old, ids_new)
        self.assertEqual(types_old, types_new)

        type_0_cols_new = [colloid for colloid in colloids_new if colloid.type == 0]
        type_1_cols_new = [colloid for colloid in colloids_new if colloid.type == 1]
        type_0_cols_old = [colloid for colloid in colloids_old if colloid.type == 0]
        type_1_cols_old = [colloid for colloid in colloids_old if colloid.type == 1]

        self.assertEqual(len(type_0_cols_new), len(type_0_cols_old))
        self.assertEqual(len(type_1_cols_new), len(type_1_cols_old))

    def test_add_source(self):
        # add a different type of colloids and check if the number of colloids is
        # correct
        # also check if the ids and the types are correct

        self.system_runner.add_source(
            pos=self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
            source_particle_type=2,
        )

        # check if it is the correct number of colloids
        self.assertEqual(len(self.system_runner.colloids), 21)

        colloids_old = copy.deepcopy(self.system_runner.colloids)
        ids_old = [colloid.id for colloid in colloids_old]
        types_old = [colloid.type for colloid in colloids_old]

        self.system_runner.reset_system()
        colloids_new = copy.deepcopy(self.system_runner.colloids)
        ids_new = [colloid.id for colloid in colloids_new]
        types_new = [colloid.type for colloid in colloids_new]

        self.assertEqual(len(self.system_runner.colloids), 21)
        self.assertEqual(ids_old, ids_new)
        self.assertEqual(types_old, types_new)

        type_0_cols_new = [colloid for colloid in colloids_new if colloid.type == 0]
        type_1_cols_new = [colloid for colloid in colloids_new if colloid.type == 1]
        type_2_cols_new = [colloid for colloid in colloids_new if colloid.type == 2]
        type_0_cols_old = [colloid for colloid in colloids_old if colloid.type == 0]
        type_1_cols_old = [colloid for colloid in colloids_old if colloid.type == 1]
        type_2_cols_old = [colloid for colloid in colloids_old if colloid.type == 2]

        self.assertEqual(len(type_0_cols_new), len(type_0_cols_old))
        self.assertEqual(len(type_1_cols_new), len(type_1_cols_old))
        self.assertEqual(len(type_2_cols_new), len(type_2_cols_old))

    def test_add_rod(self):
        self.system_runner.add_rod(
            rod_center=self.ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
            rod_length=self.ureg.Quantity(30.0, "micrometer"),
            rod_thickness=self.ureg.Quantity(4.0, "micrometer"),
            rod_start_angle=0.0,
            n_particles=10,
            friction_trans=0.1,
            friction_rot=0.1,
            rod_particle_type=3,
        )

        colloids_old = copy.deepcopy(self.system_runner.colloids)
        ids_old = [colloid.id for colloid in colloids_old]
        types_old = [colloid.type for colloid in colloids_old]

        # check if it is the correct number of colloids
        self.assertEqual(len(self.system_runner.colloids), 31)
        self.system_runner.reset_system()

        colloids_new = copy.deepcopy(self.system_runner.colloids)
        ids_new = [colloid.id for colloid in colloids_new]
        types_new = [colloid.type for colloid in colloids_new]

        self.assertEqual(len(self.system_runner.colloids), 31)
        self.assertEqual(ids_old, ids_new)
        self.assertEqual(types_old, types_new)

        type_0_cols_new = [colloid for colloid in colloids_new if colloid.type == 0]
        type_1_cols_new = [colloid for colloid in colloids_new if colloid.type == 1]
        type_2_cols_new = [colloid for colloid in colloids_new if colloid.type == 2]
        type_3_cols_new = [colloid for colloid in colloids_new if colloid.type == 3]
        type_0_cols_old = [colloid for colloid in colloids_old if colloid.type == 0]
        type_1_cols_old = [colloid for colloid in colloids_old if colloid.type == 1]
        type_2_cols_old = [colloid for colloid in colloids_old if colloid.type == 2]
        type_3_cols_old = [colloid for colloid in colloids_old if colloid.type == 3]

        self.assertEqual(len(type_0_cols_new), len(type_0_cols_old))
        self.assertEqual(len(type_1_cols_new), len(type_1_cols_old))
        self.assertEqual(len(type_2_cols_new), len(type_2_cols_old))
        self.assertEqual(len(type_3_cols_new), len(type_3_cols_old))

    def test_full_sim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.rl_simulation(
                outfolder,
            )


if __name__ == "__main__":
    ut.main()
