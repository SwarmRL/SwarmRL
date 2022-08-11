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
            box_length=ureg.Quantity(10, "micrometer"),
            time_step=ureg.Quantity(0.0001, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=3, out_folder=temp_dir, write_chunk_size=1
            )

            coll_type = 1
            runner.add_colloids(
                5,
                radius_colloid=ureg.Quantity(1.0, "micrometer"),
                random_placement_center=ureg.Quantity(
                    np.array(3 * [5.0]), "micrometer"
                ),
                random_placement_radius=ureg.Quantity(4, "micrometer"),
                type_colloid=coll_type,
            )
            with self.assertRaises(ValueError):
                runner.add_confining_walls(coll_type)
            runner.add_confining_walls(coll_type + 1)
            assert len(runner.system.constraints) == 2 * runner.n_dims

            const_force = dummy_models.ConstForce(force=10)
            runner.integrate(300, const_force)

            # without walls, the colloids would leave the primary box
            poss = runner.get_particle_data()["Unwrapped_Positions"]
            assert np.all(poss < 10)


if __name__ == "__main__":
    ut.main()
