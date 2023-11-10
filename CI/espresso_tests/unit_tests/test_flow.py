import tempfile
import unittest as ut

import numpy as np
import pint

from swarmrl.engine import espresso
from swarmrl.models import dummy_models
from swarmrl.utils import utils


class FlowTest(ut.TestCase):
    def test_flow_and_potential(self):
        """
        Set up a 2d simulation with flow and boundaries.
        Also introduces the tricks needed to perform physcially meaningful simulations.
        """
        ureg = pint.UnitRegistry()
        langevin_friction_scale = 1e-6  # to avoid double-friction, see below
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-3, "pascal * second"),
            WCA_epsilon=ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(300, "kelvin") / langevin_friction_scale,
            box_length=ureg.Quantity(100, "micrometer"),
            time_step=ureg.Quantity(0.1, "second") / 100,
            time_slice=ureg.Quantity(0.01, "second"),
            write_interval=ureg.Quantity(5, "second"),
            thermostat_type="langevin",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, out_folder=temp_dir, write_chunk_size=1, n_dims=2
            )

            # create a 2d flowfield. Usually, this would be done in a
            # separate (lattice Boltzmann) simulation
            # note that swarmrl only supports cubic simulation boxes
            n_cells = 100
            box_l = params.box_length.m_as("sim_length")
            agrid_SI = params.box_length / n_cells
            agrid = box_l / n_cells
            xs = np.linspace(
                0.5 * agrid, box_l - 0.5 * agrid, num=n_cells, endpoint=True
            )
            ys = xs
            xv, yv = np.meshgrid(xs, ys)

            flow = 3 * np.stack(
                [np.ones_like(xv), 0.1 * np.cos(yv) * np.sin(xv), np.zeros_like(xv)],
                axis=2,
            )
            flow = ureg.Quantity(flow, "micrometer/second")

            # create binary mask of boundary: circular domain
            rs = np.sqrt((xv - box_l / 2.0) ** 2 + (yv - box_l / 2.0) ** 2)
            boundary_mask = rs > box_l / 2.0 - agrid

            # Before adding the flowfield, we need to know the coupling gamma
            coll_radius = ureg.Quantity(1, "micrometer")
            gamma = 6 * np.pi * params.fluid_dyn_viscosity * coll_radius

            # flowfield must be 3d, so we add z-axis
            runner.add_flowfield(
                flow[:, :, np.newaxis, :],
                gamma,
                utils.convert_array_of_pint_to_pint_of_array(
                    [agrid_SI, agrid_SI, params.box_length], ureg
                ),
            )

            """
            force at the boundary comes from interpolation of step potential
            across one agrid, so force=potential_jump/agrid
            choose potential jump such that force is
            larger than swim force + drag force
            """
            swim_vel = ureg.Quantity(5, "micrometer/second")
            swim_force = gamma * swim_vel
            potential = (
                2 * (swim_force + np.max(flow) * gamma) * agrid_SI * boundary_mask
            )
            runner.add_external_potential(
                potential[:, :, np.newaxis],
                utils.convert_array_of_pint_to_pint_of_array(
                    [agrid_SI, agrid_SI, params.box_length], ureg
                ),
            )

            """
            Flow constraint introduces friction.
            Thermostat has friction and noise.
            To avoid double-friction, we need to get rid of the thermostat
            friction while retaining the noise. Langevin noise scales with kT*gamma,
            so we must keep the product constant while reducing the gamma.
            This is why the temperature in MDParams is increased.
            For rotation, the flow does not introduce additional friction,
            so we need to cope with the reduced temperature:
            1) The thermostat rotational friction must be increased
            to get the correct noise amplitude.
            2) All torques (e.g. the ones from actions) must be increased
            to reflect the increased rotational friction.
            """

            gamma_trans_reduced = gamma * langevin_friction_scale
            gamma_rot_increased = (
                8
                * np.pi
                * params.fluid_dyn_viscosity
                * coll_radius**3
                / langevin_friction_scale
            )
            active_ang_vel = ureg.Quantity(4 * np.pi / 180, "1/second")

            """
            We can't do actual langevin dynamics of the colloids
            (momentum relaxation timescale is way too small to resolve).
            By increasing the particle mass, we bring the momentum relaxation
            closer to the timescales we are interested in (and can resolve).
            However, we cannot increase too much, because at some point
            there will be inertial effects. Here, we choose 0.01 seconds such that
            the momentum relaxation is much faster than the change in actions.
            For production simulations you should consider using an even smaller
            value to actually have the timescales separated.
            """
            target_momentum_relaxation_timescale = ureg.Quantity(0.01, "second")
            partcl_mass = gamma * target_momentum_relaxation_timescale
            partcl_rinertia = gamma_rot_increased * target_momentum_relaxation_timescale

            runner.add_colloids(
                10,
                coll_radius,
                ureg.Quantity([50, 50, 0], "micrometer"),
                ureg.Quantity(30, "micrometer"),
                gamma_translation=gamma_trans_reduced,
                gamma_rotation=gamma_rot_increased,
                mass=partcl_mass,
                rinertia=utils.convert_array_of_pint_to_pint_of_array(
                    3 * [partcl_rinertia], ureg
                ),
            )

            active_force = swim_force.m_as("sim_force")
            active_torque = (active_ang_vel * gamma_rot_increased).m_as("sim_torque")

            model = dummy_models.ConstForceAndTorque(
                active_force, [0, 0, active_torque]
            )

            runner.integrate(1000, force_model=model)

            # particles should be advected to the right until they hit the boundary
            poss = runner.get_particle_data()["Unwrapped_Positions"]
            np.testing.assert_array_less(box_l / 2.0, np.mean(poss[:, 0]))
            np.testing.assert_array_less(poss[:, 0], box_l)


if __name__ == "__main__":
    ut.main()
