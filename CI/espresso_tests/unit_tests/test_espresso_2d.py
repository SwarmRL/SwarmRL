import tempfile
import unittest as ut

import espressomd
import numpy as np
import pint

import swarmrl.utils
from swarmrl.agents import dummy_models
from swarmrl.engine import espresso
from swarmrl.force_functions import ForceFunction


def assertNotArrayAlmostEqual(arr0, arr1, atol=1e-6):
    with np.testing.assert_raises(Exception):
        np.testing.assert_array_almost_equal(arr0, arr1, atol=atol)


class EspressoTest2D(ut.TestCase):
    """
    Tests the implementation of 2d motion in swarmrl's espresso engine.
    Isotropic and anisotropic particles are tested separately.
    Indirectly, we also test the possibility to recycle an espressomd.System
    for multiple independent engines.
    """

    system = espressomd.System(box_l=[1, 2, 3])

    def test_isotropic(self):
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(1e-20, "joule"),
            temperature=ureg.Quantity(300, "kelvin"),
            box_length=ureg.Quantity(3 * [10000], "micrometer"),
            time_step=ureg.Quantity(0.05, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params,
                n_dims=2,
                out_folder=temp_dir,
                write_chunk_size=1,
                system=self.system,
            )
            self.assertListEqual(runner.colloids, [])

            n_colloids = 14
            runner.add_colloids(
                n_colloids,
                ureg.Quantity(1, "micrometer"),
                ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                ureg.Quantity(100, "micrometer"),
                type_colloid=3,
            )
            self.assertEqual(len(runner.system.part.all()), n_colloids)
            np.testing.assert_allclose(runner.system.part.all().pos[:, 2], 0)

            part_data = runner.get_particle_data()

            directors = part_data["Directors"]
            directors_z = directors[:, 2]
            np.testing.assert_array_almost_equal(
                directors_z, np.zeros_like(directors_z)
            )

            no_force = dummy_models.ConstForce(force=0)
            # brownian motion in xy-plane and rotation around z
            force_fn = ForceFunction({"3": no_force})
            runner.integrate(10, force_fn)
            part_data_new = runner.get_particle_data()
            directors_new = part_data_new["Directors"]
            directors_new_z = directors_new[:, 2]
            np.testing.assert_array_almost_equal(
                directors_new_z, np.zeros_like(directors_new_z)
            )

            assertNotArrayAlmostEqual(directors, directors_new)

            # test rotation from force model
            orientation = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])
            rotator = dummy_models.ToConstDirection(orientation)
            force_fn = ForceFunction({"3": rotator})
            runner.manage_forces(force_fn)
            runner.system.integrator.run(0)
            part_data_rot = runner.get_particle_data()
            directors_rot = part_data_rot["Directors"]
            for dir_ in directors_rot:
                np.testing.assert_array_almost_equal(dir_, orientation)

    def test_anisotropic(self):
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=0.01 * ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            # small epsilon to make gay-berne not significantly attractive
            temperature=ureg.Quantity(
                0, "kelvin"
            ),  # to test the friction without noise
            box_length=ureg.Quantity(3 * [10000], "micrometer"),
            time_step=ureg.Quantity(0.01, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params,
                n_dims=2,
                out_folder=temp_dir,
                write_chunk_size=1,
                system=self.system,
            )

            # radius for sphere comparison.
            # no semiaxis of the ellipsoid will have this radius
            radius = ureg.Quantity(1, "micrometer")
            aspect_ratio = 3  # prolate because >1

            # calculate semiaxes via volume equivalent
            equatorial_semiaxis = radius / np.cbrt(aspect_ratio)
            axial_semiaxis = equatorial_semiaxis * aspect_ratio

            (
                gamma_trans_ax,
                gamma_trans_eq,
            ) = swarmrl.utils.calc_ellipsoid_friction_factors_translation(
                axial_semiaxis, equatorial_semiaxis, params.fluid_dyn_viscosity
            )
            (
                gamma_rot_ax,
                gamma_rot_eq,
            ) = swarmrl.utils.calc_ellipsoid_friction_factors_rotation(
                axial_semiaxis, equatorial_semiaxis, params.fluid_dyn_viscosity
            )

            gamma_trans = swarmrl.utils.convert_array_of_pint_to_pint_of_array(
                [gamma_trans_eq, gamma_trans_eq, gamma_trans_ax], ureg
            )
            gamma_rot = swarmrl.utils.convert_array_of_pint_to_pint_of_array(
                [gamma_rot_eq, gamma_rot_eq, gamma_rot_ax], ureg
            )

            runner.add_colloid_on_point(
                equatorial_semiaxis,
                ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                init_direction=np.array([0, 1, 0]),
                gamma_translation=gamma_trans,
                gamma_rotation=gamma_rot,
                aspect_ratio=aspect_ratio,
            )

            es_partcl = runner.system.part.by_id(0)
            dummy_model = dummy_models.ConstForce(force=0)
            force_fn = ForceFunction({"0": dummy_model})
            es_partcl.ext_force = [
                1,
                1,
                0,
            ]  # y is the axial, x the equatorial direction
            runner.integrate(10, force_fn)
            np.testing.assert_almost_equal(
                np.copy(es_partcl.v),
                [
                    1 / gamma_trans_eq.m_as("sim_force/sim_velocity"),
                    1 / gamma_trans_ax.m_as("sim_force/sim_velocity"),
                    0,
                ],
            )

            es_partcl.ext_force = [0, 0, 0]
            const_torque = dummy_models.ConstTorque(
                torque=np.array([0, 0, 1])
            )  # equatorial rotation
            force_fn = ForceFunction({"0": const_torque})
            runner.integrate(10, force_fn)
            np.testing.assert_almost_equal(
                np.copy(es_partcl.omega_lab),
                [0, 0, 1 / gamma_rot_eq.m_as("sim_torque/sim_angular_velocity")],
            )

            # no need to check axial friction because rotation around
            # the symmetry-axis is disabled anyways.


if __name__ == "__main__":
    ut.main()
