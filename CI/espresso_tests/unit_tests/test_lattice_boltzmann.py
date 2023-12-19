import tempfile
import unittest as ut

import numpy as np
import pint

import swarmrl.utils
from swarmrl.agents import dummy_models
from swarmrl.engine import espresso
from swarmrl.force_functions import ForceFunction


class LatticeBoltzmannTest(ut.TestCase):
    def test_class(self):
        ureg = pint.UnitRegistry()
        # large box for no particle interaction
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-3, "pascal * second"),
            temperature=ureg.Quantity(0, "kelvin"),
            box_length=ureg.Quantity(20, "micrometer"),
            time_step=ureg.Quantity(0.0002, "second"),
            write_interval=ureg.Quantity(10, "second"),
            thermostat_type="langevin",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, out_folder=temp_dir, write_chunk_size=1, seed=42
            )

            agrid = ureg.Quantity(1, "micrometer")
            n_cells = (
                np.round(ureg.Quantity(3 * [20], "micrometer") / agrid)
                .m_as("dimensionless")
                .astype(int)
            )

            # poiseuille flow along x
            boundaries = np.zeros(n_cells, dtype=bool)
            boundaries[:, :, 0] = True
            boundaries[:, :, -1] = True

            target_mean_vel = ureg.Quantity(0.5, "micrometer/second")

            channel_height = params.box_length - 2 * agrid
            dyn_visc = params.fluid_dyn_viscosity

            # scale up density to not have extremely small Re
            # (and small relaxation time)
            target_reynolds = 1e-1
            density = target_reynolds * dyn_visc / (target_mean_vel * channel_height)

            # https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation
            pressure_gradient = (
                target_mean_vel * channel_height * 12 * dyn_visc / channel_height**3
            )
            pressure_gradient = swarmrl.utils.convert_array_of_pint_to_pint_of_array(
                [
                    pressure_gradient,
                    ureg.Quantity(0, "N/m**3"),
                    ureg.Quantity(0, "N/m**3"),
                ],
                ureg,
            )

            lbf = runner.add_lattice_boltzmann(
                agrid=agrid,
                ext_force_density=pressure_gradient,
                fluid_density=density,
                boundary_mask=boundaries,
            )
            partcl_density = 10 * density
            partcl_radius = ureg.Quantity(0.1, "micrometer")  # smaller than agrid!
            runner.add_colloids(
                50,
                type_colloid=0,
                radius_colloid=partcl_radius,
                mass=partcl_density * 4 / 3 * np.pi * partcl_radius**3,
                rinertia=swarmrl.utils.convert_array_of_pint_to_pint_of_array(
                    3 * [2.0 / 5.0 * partcl_density * 4 / 3 * np.pi * partcl_radius**5],
                    ureg,
                ),
            )

            force_model = dummy_models.ConstForce(0.0)
            force_fn = ForceFunction({"0": force_model})

            runner.integrate(25, force_fn)

            flowfield = np.copy(lbf[:, :, :].velocity)

            mean_vel = np.mean(flowfield[:, :, 1:-1, 0])
            np.testing.assert_allclose(
                mean_vel, target_mean_vel.m_as("micrometer/second"), rtol=1e-2
            )

            vels = runner.get_particle_data()["Velocities"]
            # no noise, no force model => particles just move with the fluid
            # large error margin allowed bcs this is an average over just 50 particles
            np.testing.assert_allclose(
                np.mean(vels, axis=0),
                [target_mean_vel.m_as("micrometer/second"), 0, 0],
                atol=0.1,
            )


if __name__ == "__main__":
    ut.main()
