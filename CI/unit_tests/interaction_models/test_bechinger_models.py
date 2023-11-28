import unittest as ut

import numpy as np

import swarmrl.models.bechinger_models
import swarmrl.models.interaction_model as int_mod


class TestLavergne(ut.TestCase):
    def setUp(self) -> None:
        self.act_force = 1.234
        self.vision_half_angle = np.pi / 4.0
        self.perception_threshold = 0.5

        self.force_model = swarmrl.models.bechinger_models.Lavergne2019(
            vision_half_angle=self.vision_half_angle,
            act_force=self.act_force,
            perception_threshold=self.perception_threshold,
        )

    def test_force(self):
        orientation = np.array([1, 0, 0])
        test_coll = int_mod.Colloid(pos=np.array([0, 0, 0]), director=orientation, id=1)
        coll_front = int_mod.Colloid(
            pos=np.array([100, 0, 0]), director=orientation, id=2
        )
        coll_back = int_mod.Colloid(
            pos=np.array([-0.01, 0, 0]), director=orientation, id=3
        )
        coll_side = int_mod.Colloid(
            pos=np.array([0, 0.01, 0]), director=orientation, id=4
        )

        colloids = [test_coll, coll_front, coll_back, coll_side]

        action = self.force_model.calc_action(colloids)
        print(action)
        force_is = action[0].force
        # front colloid too far, back not visible
        self.assertAlmostEqual(force_is, 0)

        coll_front_close = int_mod.Colloid(
            pos=np.array([0.1, 0, 0]), director=orientation, id=5
        )
        colloids.append(coll_front_close)

        action = self.force_model.calc_action(colloids)
        force_is = action[0].force
        # front close -> activity along orientation
        self.assertAlmostEqual(force_is, self.act_force)


class TestBaeuerle(ut.TestCase):
    def setUp(self) -> None:
        self.act_force = 1.234
        self.act_torque = 2.345
        self.vision_half_angle = np.pi / 4.0
        self.detection_radius_position = 1.1
        self.detection_radius_orientation = 0.5
        self.angular_deviation = np.pi / 8.0

        self.force_model = swarmrl.models.bechinger_models.Baeuerle2020(
            act_force=self.act_force,
            act_torque=self.act_torque,
            detection_radius_orientation=self.detection_radius_orientation,
            detection_radius_position=self.detection_radius_position,
            vision_half_angle=self.vision_half_angle,
            angular_deviation=self.angular_deviation,
        )

    def test_torque(self):
        test_coll = int_mod.Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), id=1
        )
        front_coll = int_mod.Colloid(
            pos=np.array([1, 0.1, 0]), director=np.array([0, 1, 0]), id=2
        )
        front_close_coll = int_mod.Colloid(
            pos=np.array([0.2, 0.1, 0]), director=np.array([0, -1, 0]), id=3
        )
        front_far_coll = int_mod.Colloid(
            pos=np.array([10, 0, 0]), director=np.array([0, 1, 0]), id=4
        )
        side_coll = int_mod.Colloid(
            pos=np.array([0, 0.1, 0]), director=np.array([0, 1, 0]), id=5
        )

        colloids = [test_coll, front_coll, front_close_coll, front_far_coll, side_coll]

        action = self.force_model.calc_action(colloids)
        torque = action[0].torque
        torque_norm = np.linalg.norm(torque)

        # torque has maximum value of axt_torque
        self.assertGreater(self.act_torque, torque_norm)
        # com determied by front_coll and front_close_coll
        # orientation determined by front_close_coll

        # force must be to the right of the com (coll_front_close point to -y)
        self.assertGreater(0, torque[2])


class TestUtils(ut.TestCase):
    def test_coll_in_vision(self):
        test_coll = int_mod.Colloid(
            pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), id=1
        )
        front_coll = int_mod.Colloid(
            pos=np.array([1.1, 0, 0]), director=np.array([1, 0, 0]), id=2
        )
        front_far_coll = int_mod.Colloid(
            pos=np.array([100, 0, 0]), director=np.array([1, 0, 0]), id=3
        )
        side_coll = int_mod.Colloid(
            pos=np.array([0, 0.2, 0]), director=np.array([1, 0, 0]), id=4
        )
        slight_offset_coll = int_mod.Colloid(
            pos=np.array([1, 0, 0.1]), director=np.array([1, 0, 0]), id=5
        )

        colls_in_range = swarmrl.models.bechinger_models.get_colloids_in_vision(
            test_coll,
            [front_coll, front_far_coll, side_coll, slight_offset_coll],
            vision_half_angle=np.pi / 4.0,
            vision_range=10,
        )
        colls_shouldbe = [front_coll, slight_offset_coll]

        self.assertEqual(len(colls_in_range), len(colls_shouldbe))
        for coll in colls_shouldbe:
            self.assertIn(coll, colls_in_range)


if __name__ == "__main__":
    ut.main()
