import tempfile
import unittest as ut
import numpy as np
import pint

from swarmrl.engine import espresso
from swarmrl.models import dummy_models


class EspressoTest2D(ut.TestCase):
    def assertNotArrayAlmostEqual(self, arr0, arr1):
        with self.assertRaises(Exception):
            np.testing.assert_array_almost_equal(arr0, arr1)

    def test_class(self):
        """
        Sadly, espresso systems are global so we have to do all tests on only one object
        """
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            n_colloids=10,
            ureg=ureg,
            colloid_radius=ureg.Quantity(1, "micrometer"),
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(1e-20, "joule"),
            colloid_density=ureg.Quantity(2.65, "gram / centimeter**3"),
            temperature=ureg.Quantity(300, "kelvin"),
            box_length=ureg.Quantity(10000, "micrometer"),
            time_step=ureg.Quantity(0.05, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=2, out_folder=temp_dir, write_chunk_size=1
            )
            self.assertListEqual(runner.colloids, [])

            runner.setup_simulation()
            self.assertEqual(len(runner.system.part[:]), params.n_colloids)

            part_data = runner.get_particle_data()

            directors = part_data["Directors"]
            directors_z = directors[:, 2]
            np.testing.assert_array_almost_equal(
                directors_z, np.zeros_like(directors_z)
            )

            no_force = dummy_models.ConstForce(force=0)
            # brownian motion in xy-plane and rotation around z
            runner.integrate(10, no_force)
            part_data_new = runner.get_particle_data()
            directors_new = part_data_new["Directors"]
            directors_new_z = directors_new[:, 2]
            np.testing.assert_array_almost_equal(
                directors_new_z, np.zeros_like(directors_new_z)
            )

            self.assertNotArrayAlmostEqual(directors, directors_new)

            # test rotation from force model
            orientation = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])
            rotator = dummy_models.ToConstDirection(orientation)
            runner.params.steps_per_slice = (
                0  # bad hack: do not integrate, just update the new direction
            )
            runner.integrate(1, rotator)
            part_data_rot = runner.get_particle_data()
            directors_rot = part_data_rot["Directors"]
            for dir_ in directors_rot:
                np.testing.assert_array_almost_equal(dir_, orientation)


if __name__ == "__main__":
    ut.main()
