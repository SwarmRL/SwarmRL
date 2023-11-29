import pathlib as pl
import tempfile
import unittest as ut

import h5py
import numpy as np
import pint

import swarmrl.utils
from swarmrl.agents import dummy_models
from swarmrl.engine import espresso
from swarmrl.force_functions import ForceFunction


class EspressoTest(ut.TestCase):
    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError(f"File does not exist: {path}")

    def test_class(self):
        """
        Sadly, espresso systems are global so we have to do all tests on only one object
        """
        ureg = pint.UnitRegistry()
        # large box for no particle interaction
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-3, "pascal * second"),
            WCA_epsilon=ureg.Quantity(1e-20, "joule"),
            temperature=ureg.Quantity(0, "kelvin"),
            box_length=ureg.Quantity(1000, "micrometer"),
            time_step=ureg.Quantity(0.05, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, out_folder=temp_dir, write_chunk_size=1
            )
            swarmrl.utils.setup_swarmrl_logger(f"{temp_dir}/simulation_log.log")
            self.assertListEqual(runner.colloids, [])

            n_colloids = [2, 3]
            coll_types = [1, 2]
            coll_radius = ureg.Quantity(1, "micrometer")
            box_center = ureg.Quantity(np.array(3 * [500]), "micrometer")
            init_radius = ureg.Quantity(500, "micrometer")
            runner.add_colloids(
                n_colloids[0],
                coll_radius,
                box_center,
                init_radius,
                type_colloid=coll_types[0],
            )

            runner.add_colloids(
                n_colloids[1],
                coll_radius,
                box_center,
                init_radius,
                type_colloid=coll_types[1],
            )

            self.assertEqual(len(runner.system.part.all()), sum(n_colloids))
            self.assertEqual(
                len(runner.system.part.select(type=coll_types[1])), n_colloids[1]
            )

            part_data_old = runner.get_particle_data()
            # rotate all particles along one axis
            direc = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])
            rotator = dummy_models.ToConstDirection(direc)
            force_fn = ForceFunction({"1": rotator, "2": rotator})
            runner.integrate(1, force_fn)
            runner.system.time = 0.0
            directors = runner.get_particle_data()["Directors"]
            for dir_ in directors:
                np.testing.assert_array_almost_equal(dir_, direc)

            n_slices = 15
            force = 1.234
            force_model = dummy_models.ConstForce(force)
            force_fn = ForceFunction({"1": force_model, "2": force_model})
            runner.integrate(n_slices, force_fn)

            # save_current_state_to_file
            runner._update_traj_holder()  # take the last data
            runner.write_idx += 1  # just to be correct
            runner._write_traj_chunk_to_file()
            # clear the traj_holder after finalize just in case someone keeps
            # on integrating after finalize -> no value is writen twice.
            # But one value will be written possibly before the determined time interval
            for val in runner.traj_holder.values():
                val.clear()

            self.assertIsFile(f"{temp_dir}/trajectory.hdf5")
            self.assertIsFile(f"{temp_dir}/simulation_log.log")

            part_data_new = runner.get_particle_data()
            time_new = runner.system.time

            gamma_trans, gamma_rot = runner.get_friction_coefficients(coll_types[0])
            new_vel = part_data_new["Velocities"]
            new_vel_shouldbe = force * direc / gamma_trans
            for vel in new_vel:
                np.testing.assert_array_almost_equal(vel, new_vel_shouldbe)

            old_pos = part_data_old["Unwrapped_Positions"]
            new_pos = part_data_new["Unwrapped_Positions"]

            np.testing.assert_allclose(old_pos + time_new * new_vel, new_pos)

            # check interactions after first integrate call
            wca_params = runner.system.non_bonded_inter[
                coll_types[0], coll_types[1]
            ].wca.get_params()
            cutoff = wca_params["sigma"] * 2 ** (1 / 6)
            np.testing.assert_allclose(cutoff, 2 * coll_radius.m_as("sim_length"))

            with h5py.File(f"{temp_dir}/trajectory.hdf5", "r") as h5_file:
                part_group = h5_file["colloids"]
                np.testing.assert_array_almost_equal(
                    part_group["Unwrapped_Positions"][-1, :, :], new_pos
                )
                np.testing.assert_array_almost_equal(
                    part_group["Velocities"][-1, :, :], new_vel
                )
                self.assertAlmostEqual(part_group["Times"][-1, 0, 0], time_new)
                self.assertSetEqual(
                    set(part_group.keys()),
                    {
                        "Types",
                        "Ids",
                        "Unwrapped_Positions",
                        "Velocities",
                        "Times",
                        "Directors",
                    },
                )
            const_force = ureg.Quantity(np.array([0.1, 0.2, 0.3]), "newton")
            runner.add_const_force_to_colloids(const_force, coll_types[0])

            type_0_colls = runner.system.part.select(type=coll_types[0])
            type_1_colls = runner.system.part.select(type=coll_types[1])

            assert np.all(type_0_colls.ext_force > 0)
            assert np.all(type_1_colls.ext_force == 0)


if __name__ == "__main__":
    ut.main()
