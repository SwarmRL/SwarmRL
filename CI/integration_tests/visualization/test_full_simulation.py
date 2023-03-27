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
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(293, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(293, "kelvin"),
            box_length=ureg.Quantity(1000, "micrometer"),
            time_slice=ureg.Quantity(0.2, "second"),  # model timestep
            time_step=ureg.Quantity(0.2, "second") / 5,  # integrator timestep
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
            "sim_duration": ureg.Quantity(0.1, "minute"),
            "seed": self.seed,
        }


        rod_params_type = {
            "rod_center": ureg.Quantity([500,500] "micrometer"),
            "rod_length": ureg.Quantity(100, "micrometer"),
            "rod_thickness": ureg.Quantity(100/59, "micrometer"),
            "rod_start_angle": 0,
            "n_particles": 59,
            "friction_trans": friction_trans,
            "friction_rot": friction_rot,
            "rod_particle_type": args.rod_particle_type,
            "rod_fixed": args.rod_fixed,
            "rod_center_part_id": 42,  # gets calculated
            "rod_border_parts_id": [42, 42],  # gets calculated
            "rod_break_ang_vel": args.rod_break_ang_vel,
            "rod_break": args.rod_break
        }

        # from now on, no new parameters are introduced

        system_runner = espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=run_params["seed"],
            out_folder=outfolder,
            write_chunk_size=1000,
        )

        coll_type = 0
        system_runner.add_colloids(
            run_params["n_colloids"],
            ureg.Quantity(3.15, "micrometer"),
            ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
            ureg.Quantity(60, "micrometer"),
            type_colloid=coll_type,
        )

        system_runner.add_rod(
                self.rod_params_type["rod_center"],
                self.rod_params_type["rod_length"],
                self.rod_params_type["rod_thickness"],
                self.rod_params_type["rod_start_angle"],
                self.rod_params_type["n_particles"],
                ftrans,
                froteq,
                self.rod_params_type["rod_particle_type"],
                self.rod_params_type["rod_fixed"]
            )

        gamma, gamma_rot = system_runner.get_friction_coefficients(coll_type)
        target_vel = model_params["target_vel_SI"].m_as("sim_velocity")
        act_force = target_vel * gamma
        target_ang_vel = model_params["target_ang_vel_SI"].m_as("1 / sim_time")
        act_torque = target_ang_vel * gamma_rot

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



    def test(self):
        
            self.assertEqual(n_timesteps, np.shape(positions)[0])
            self.assertEqual(n_timesteps, np.shape(directors)[0])
            self.assertEqual(n_colloids, np.shape(positions)[1])
            self.assertEqual(n_colloids, np.shape(directors)[1])
            self.assertLessEqual(positions[0, 0, 0], box_length)


if __name__ == "__main__":
    ut.main()
