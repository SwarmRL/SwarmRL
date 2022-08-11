import tempfile
import unittest as ut

import numpy as np
import pint

from swarmrl.engine import espresso
from swarmrl.models import dummy_models


def assertNotArrayAlmostEqual(arr0, arr1, atol=1e-6):
    with np.testing.assert_raises(Exception):
        np.testing.assert_array_almost_equal(arr0, arr1, atol=atol)


class EspressoTest2D(ut.TestCase):
    def test_class(self):
        """
        Sadly, espresso systems are global so we have to do all tests on only one object
        """
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(1e-20, "joule"),
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
            runner.integrate(10, no_force)
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
