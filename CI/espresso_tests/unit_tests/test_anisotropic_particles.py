import tempfile
import unittest as ut

import numpy as np
import pint

import swarmrl.utils
from swarmrl.engine import espresso
from swarmrl.models import dummy_models


class EspressoTestAnisotropic(ut.TestCase):
    def test_class(self):
        """
        Sadly, espresso systems are global so we have to do all tests on only one object
        """
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=0.01 * ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            # small epsilon to make gay-berne not significantly attractive
            temperature=ureg.Quantity(
                0, "kelvin"
            ),  # to test the friction without noise
            box_length=ureg.Quantity(10000, "micrometer"),
            time_step=ureg.Quantity(0.01, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=2, out_folder=temp_dir, write_chunk_size=1
            )

            # radius for sphere comparison.
            # no semiaxis of the ellipsoid will have this radius
            radius = ureg.Quantity(1, "micrometer")
            aspect_ratio = 1 / 3  # prolate because >1

            # calculate semiaxes via volume equivalent
            equatorial_semiaxis = radius / np.cbrt(aspect_ratio)
            axial_semiaxis = equatorial_semiaxis * aspect_ratio

            gamma_trans_ax, gamma_trans_eq = (
                swarmrl.utils.calc_ellipsoid_friction_factors_translation(
                    axial_semiaxis, equatorial_semiaxis, params.fluid_dyn_viscosity
                )
            )
            gamma_rot_ax, gamma_rot_eq = (
                swarmrl.utils.calc_ellipsoid_friction_factors_rotation(
                    axial_semiaxis, equatorial_semiaxis, params.fluid_dyn_viscosity
                )
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

            es_partcl.ext_force = [
                1,
                1,
                0,
            ]  # y is the axial, x the equatorial direction
            runner.integrate(10, dummy_model)
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
            runner.integrate(10, const_torque)
            np.testing.assert_almost_equal(
                np.copy(es_partcl.omega_lab),
                [0, 0, 1 / gamma_rot_eq.m_as("sim_torque/sim_angular_velocity")],
            )

            # no need to check axial friction because rotation around
            # the symmetry-axis is disabled anyways.


if __name__ == "__main__":
    ut.main()
