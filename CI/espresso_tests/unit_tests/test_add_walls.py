import tempfile
import unittest as ut

import numpy as np
import pint

from swarmrl.agents import dummy_models
from swarmrl.engine import espresso
from swarmrl.force_functions import ForceFunction


class AddWalls(ut.TestCase):
    def test_class(self):
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=0.1 * ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(300, "kelvin"),
            box_length=ureg.Quantity(100, "micrometer"),
            time_step=ureg.Quantity(0.005, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=2, out_folder=temp_dir, write_chunk_size=1
            )

            coll_type = 1
            runner.add_colloids(
                5,
                radius_colloid=ureg.Quantity(1.0, "micrometer"),
                random_placement_center=ureg.Quantity(
                    np.array([50, 50, 0]), "micrometer"
                ),
                random_placement_radius=ureg.Quantity(4, "micrometer"),
                type_colloid=coll_type,
            )
            wall_start_point = ureg.Quantity(
                np.array(
                    [
                        [40, 40],
                        [40, 40],
                        [60, 60],
                        [60, 60],
                    ]
                ),
                "micrometer",
            )
            wall_end_point = ureg.Quantity(
                np.array(
                    [
                        [40, 60],
                        [60, 40],
                        [40, 60],
                        [60, 40],
                    ]
                ),
                "micrometer",
            )
            wall_thickness = ureg.Quantity(2, "micrometer")
            with self.assertRaises(ValueError):
                runner.add_walls(
                    wall_start_point=wall_start_point,
                    wall_end_point=wall_end_point,
                    wall_type=coll_type,
                    wall_thickness=wall_thickness,
                )
            runner.add_walls(
                wall_start_point=wall_start_point,
                wall_end_point=wall_end_point,
                wall_type=coll_type + 1,
                wall_thickness=wall_thickness,
            )

            assert len(runner.system.constraints) == 4

            const_force = dummy_models.ConstForce(force=10)
            force_fn = ForceFunction({"0": const_force})
            runner.integrate(300, force_fn)

            # without walls, the colloids would leave the primary box
            poss = runner.get_particle_data()["Unwrapped_Positions"]
            poss = np.array(poss)
            assert np.all(poss[:, :2] < 60)
            assert np.all(poss[:, :2] > 40)


if __name__ == "__main__":
    ut.main()
