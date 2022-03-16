import copy
import pickle
import tempfile
import unittest as ut

import h5py
import numpy as np
import pint
import tqdm

import swarmrl.engine.espresso as espresso
import swarmrl.models.bechinger_models as bechinger_models
from swarmrl.utils import utils


class TestFullSim(ut.TestCase):
    """
    Functional test, also to be used as an example for simulation scripts
    """

    # Parameters usually provided as commandline input
    simulation_name = "example_simulation"
    loglevel_terminal = "info"
    seed = 42

    def simulate_model(self, outfolder):
        logger = utils.setup_swarmrl_logger(
            f"{outfolder}/{self.simulation_name}.log",
            loglevel_terminal=self.loglevel_terminal,
        )
        logger.info("Starting simulation setup")

        ureg = pint.UnitRegistry()
        md_params = espresso.MDParams(
            n_colloids=10,
            ureg=ureg,
            colloid_radius=ureg.Quantity(3.15, "micrometer"),
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(293, "kelvin") * ureg.boltzmann_constant,
            colloid_density=ureg.Quantity(2.65, "gram / centimeter**3"),
            temperature=ureg.Quantity(293, "kelvin"),
            box_length=ureg.Quantity(1000, "micrometer"),
            initiation_radius=ureg.Quantity(60, "micrometer"),
            time_slice=ureg.Quantity(0.2, "second"),
            time_step=ureg.Quantity(0.2, "second") / 5,
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

        run_params = {"sim_duration": ureg.Quantity(3, "minute"), "seed": self.seed}

        # from now on, no new parameters are introduced

        system_runner = espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=run_params["seed"],
            out_folder=outfolder,
            write_chunk_size=1000,
        )

        # setup_simulation() is needed here to calculate the friction coefficients
        system_runner.setup_simulation()
        gamma = system_runner.colloid_friction_translation
        target_vel = model_params["target_vel_SI"].m_as("sim_velocity")
        act_force = target_vel * gamma

        gamma_rotation = system_runner.colloid_friction_rotation
        target_ang_vel = model_params["target_ang_vel_SI"].m_as("1 / sim_time")
        act_torque = target_ang_vel * gamma_rotation

        detection_radius_pos = model_params["detection_radius_position_SI"].m_as(
            "sim_length"
        )
        detection_radius_or = model_params["detection_radius_orientation_SI"].m_as(
            "sim_length"
        )

        force_model = bechinger_models.Baeuerle2020(
            act_force=act_force,
            act_torque=act_torque,
            detection_radius_position=detection_radius_pos,
            detection_radius_orientation=detection_radius_or,
            vision_half_angle=model_params["vision_half_angle"],
            angular_deviation=model_params["angular_deviation"],
        )

        n_slices = int(np.ceil(run_params["sim_duration"] / md_params.time_slice))
        run_params["n_slices"] = n_slices

        # ureg can sadly not be saved by pickle
        md_params_without_ureg = copy.deepcopy(md_params)
        md_params_without_ureg.ureg = None
        params_to_write = {
            "type": type(force_model),
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

        logger.info("Starting simulation")
        for _ in tqdm.tqdm(range(100)):
            system_runner.integrate(int(np.ceil(n_slices / 100)), force_model)

        system_runner.finalize()
        logger.info("Simulation completed successfully")

    def load_results(self, outfolder):
        param_fname = f"{outfolder}/params_{self.simulation_name}.pick"
        traj_fname = f"{outfolder}/trajectory.hdf5"

        with open(param_fname, "rb") as param_file:
            params = pickle.load(param_file)
        with h5py.File(traj_fname) as traj_file:
            positions = np.array(traj_file["colloids/Unwrapped_Positions"][:, :, :2])
            directors = np.array(traj_file["colloids/Directors"][:, :, :2])
            times = np.array(traj_file["colloids/Times"][:, 0, 0])

        return params, positions, directors, times

    def test_full_sim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.simulate_model(outfolder)
            params, positions, directors, times = self.load_results(outfolder)

            # just for illustration purposes let's access some data

            n_colloids = params["md_params"].n_colloids
            box_length = params["md_params"].box_length.m_as("micrometer")

            n_timesteps = len(times)
            self.assertEqual(n_timesteps, np.shape(positions)[0])
            self.assertEqual(n_timesteps, np.shape(directors)[0])
            self.assertEqual(n_colloids, np.shape(positions)[1])
            self.assertEqual(n_colloids, np.shape(directors)[1])
            self.assertLessEqual(positions[0, 0, 0], box_length)


if __name__ == "__main__":
    ut.main()
