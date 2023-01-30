import tempfile
import unittest as ut

import numpy as np
import pint

from swarmrl.engine import espresso
from swarmrl.models import dummy_models


class RodTest(ut.TestCase):
    def test_class(self):
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=0.1 * ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(300, "kelvin"),
            box_length=ureg.Quantity(50, "micrometer"),
            time_step=ureg.Quantity(0.0001, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=2, out_folder=temp_dir, write_chunk_size=1
            )
            self.assertListEqual(runner.colloids, [])

            n_colloids = 50
            coll_rad = ureg.Quantity(1.0, "micrometer")
            runner.add_colloids(
                n_colloids,
                coll_rad,
                ureg.Quantity(np.array([25, 25, 0]), "micrometer"),
                ureg.Quantity(25, "micrometer"),
                type_colloid=0,
            )

            fric_rot = ureg.Quantity(1e-17, "newton * meter * second")
            rod_thickness = ureg.Quantity(3, "micrometer")
            center_part = runner.add_rod(
                ureg.Quantity(np.array([25, 25, 0]), "micrometer"),
                ureg.Quantity(40, "micrometer"),
                rod_thickness,
                np.pi / 2.0,
                31,
                ureg.Quantity(
                    10, "newton/ (meter / second)"
                ),  # unused value because fixed
                fric_rot,
                2,
                fixed=True,
            )

            no_force = dummy_models.ConstForce(force=0)
            runner.integrate(1, no_force)
            center_before = np.copy(center_part.pos)
            director_before = np.copy(center_part.director)
            runner.integrate(100, no_force)
            center_after = np.copy(center_part.pos)
            director_after = np.copy(center_part.director)

            np.testing.assert_allclose(center_before, center_after)
            self.assertGreater(np.linalg.norm(director_after - director_before), 1e-3)
            # check correct wca between differently sized particles
            wca_params = runner.system.non_bonded_inter[0, 2].wca.get_params()
            cutoff = wca_params["sigma"] * 2 ** (1 / 6)
            np.testing.assert_allclose(
                cutoff,
                coll_rad.m_as("sim_length") + rod_thickness.m_as("sim_length") / 2,
            )


if __name__ == "__main__":
    ut.main()
