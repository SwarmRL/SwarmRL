import tempfile
import unittest as ut

import numpy as np
import pint

from swarmrl.engine import espresso
from swarmrl.models import dummy_models


class AddColloidTest(ut.TestCase):
    def test_class(self):
        ureg = pint.UnitRegistry()
        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=0.1 * ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(0, "kelvin"),
            box_length=ureg.Quantity(100, "micrometer"),
            time_step=ureg.Quantity(0.0001, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=2, out_folder=temp_dir, write_chunk_size=1
            )

            coll_type = 2
            randomly_setup_colloids=5
            runner.add_colloids(
                randomly_setup_colloids,
                radius_colloid=ureg.Quantity(1.0, "micrometer"),
                random_placement_center=ureg.Quantity(
                    np.array([50,50,0]), "micrometer"
                ),
                random_placement_radius=ureg.Quantity(4, "micrometer"),
                type_colloid=coll_type,
            )

            position0=[20,10,0]
            directors0=np.array([1,0,0])
            type0=1
            runner.add_colloid_on_point(
                radius_colloid=ureg.Quantity(3.15,"micrometer"),
                init_position=ureg.Quantity(position0,"micrometer"),
                init_direction=directors0,
                type_colloid=type0,
            )

            position1=[80,30,0]
            directors1=np.array([1/np.sqrt(2),1/np.sqrt(2),0])
            type1=0
            runner.add_colloid_on_point(
                radius_colloid=ureg.Quantity(3.15,"micrometer"),
                init_position=ureg.Quantity(position1,"micrometer"),
                init_direction=directors1,
                type_colloid=type1,
            )

            position2=[80,30,0]
            directors2=np.array([1/np.sqrt(2),1/np.sqrt(2),0])
            type2=1
            with self.assertRaises(ValueError):
                runner.add_colloid_on_point(
                    radius_colloid=ureg.Quantity(44,"micrometer"),
                    init_position=ureg.Quantity(position2,"micrometer"),
                    init_direction=directors2,
                    type_colloid=type2,
                )

            const_force = dummy_models.ConstForce(force=0)
            runner.integrate(1, const_force)

            # without walls, the colloids would leave the primary box
            poss = runner.get_particle_data()["Unwrapped_Positions"]
            directors = runner.get_particle_data()["Directors"]
            col_types = runner.get_particle_data()["Type"]
            directors = np.array(directors)
            poss = np.array(poss)

            assert np.all(poss[:randomly_setup_colloids,:2] > 40)
            assert np.all(poss[:randomly_setup_colloids,:2] < 60)
            assert np.all(abs(poss[:randomly_setup_colloids,2])<10e-6)
            assert np.all(abs(directors[:randomly_setup_colloids,2])<10e-6)
            assert np.all(abs(np.linalg.norm(directors[:randomly_setup_colloids,:2],axis=1)-1)<10e-6)

            assert np.all(poss[5,:]==position0)
            assert np.linalg.norm(directors[5,:]-directors0)<10e-6
            assert col_types[5]==type0

            assert np.all(poss[6,:]==position1)
            assert np.linalg.norm(directors[6,:]-directors1)<10e-6
            assert col_types[6]==type1



if __name__ == "__main__":
    ut.main()
