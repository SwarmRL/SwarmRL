import unittest as ut

import numpy as np

import swarmrl.models.find_point
import swarmrl.models.interaction_model as int_mod


class TestFindPoint(ut.TestCase):
    def setUp(self) -> None:
        self.act_force = 1.234
        self.point = np.array([1, 0, 0])
        self.act_torque = 1.234

        self.force_model = swarmrl.models.find_point.FindPoint(
            act_force=self.act_force, act_torque=self.act_torque, point=self.point
        )

    def test_force(self):
        orientation = np.array([1, 0, 0])
        test_coll_front = int_mod.Colloid(
            pos=np.array([2, 0, 0]), director=orientation, id=1
        )

        colloids = [test_coll_front]

        action = self.force_model.calc_action(colloids)

        force_is = action[0].force
        # front colloid too far, back not visible
        print(force_is)
        self.assertAlmostEqual(force_is, 0)

        test_coll_behind = int_mod.Colloid(
            pos=np.array([0, 0, 0]), director=orientation, id=5
        )
        colloids.append(test_coll_behind)

        action = self.force_model.calc_action(colloids)
        force_is = action[-1].force
        # front close -> activity along orientation
        print(force_is)
        self.assertAlmostEqual(force_is, self.act_force)


if __name__ == "__main__":
    ut.main()
