import logging
import unittest as ut

import numpy as np

import swarmrl.agents.lymburn_model
from swarmrl.components.colloid import Colloid


class TestLymburnModel(ut.TestCase):
    def setUp(self):
        # Set up any necessary test data or objects
        self.force_params = {"K_a": 0, "K_r": 0, "K_h": 0, "K_f": 0, "K_p": 0}

        self.pred_params = [1000, 0.4, 0]
        self.force_model = swarmrl.agents.lymburn_model.Lymburn(
            force_params=self.force_params,
            detection_radius_position_colls=10.0,
            detection_radius_position_pred=20,
            home_pos=np.array([500, 500, 0]),
        )

        logging.basicConfig(level=logging.DEBUG)

    def test_update_force_params(self):
        self.force_model.update_force_params(K_a=1)
        self.assertEqual(self.force_model.force_params["K_a"], 1)

    def test_alignment_force(self):
        # not really sure how to test this
        self.force_model.update_force_params(K_a=1, K_r=0, K_h=0, K_f=0, K_p=0)
        coll1 = Colloid(
            pos=np.array([500.0, 500.0, 0]),
            director=np.array([1.0, 0.0, 0.0]),
            id=1,
            velocity=np.array([5.0, 0.0, 0.0]),
            type=0,
        )
        coll2 = Colloid(
            pos=np.array([505.0, 500.0, 0.0]),
            director=np.array([0.0, 1.0, 0.0]),
            id=2,
            velocity=np.array([5.0, 0.0, 0.0]),
            type=0,
        )
        action = self.force_model.calc_action([coll1, coll2])
        force_coll1 = action[0].force
        self.assertEqual(force_coll1, 0)

    def test_repulsion_force(self):
        self.force_model.update_force_params(K_a=0, K_r=1, K_h=0, K_f=0, K_p=0)

        left_coll = Colloid(
            pos=np.array([496.0, 500.0, 0]),
            director=np.array([1.0, 0.0, 0.0]),
            id=1,
            velocity=np.array([10.0, 0.0, 0.0]),
            type=0,
        )

        right_coll = Colloid(
            pos=np.array([504.0, 500.0, 0.0]),
            director=np.array([-1.0, 0.0, 0.0]),
            id=2,
            velocity=np.array([-10.0, 0.0, 0.0]),
            type=0,
        )
        far_right_coll = Colloid(  # should not be in range
            pos=np.array([600.0, 500.0, 0.0]),
            director=np.array([-1.0, 0.0, 0.0]),
            id=3,
            velocity=np.array([-10.0, 0.0, 0.0]),
            type=0,
        )
        action = self.force_model.calc_action([left_coll, right_coll, far_right_coll])

        force_left_coll = action[0].force
        force_right_coll = action[1].force
        force_far_right_coll = action[2].force

        self.assertEqual(force_left_coll, force_right_coll)
        self.assertEqual(np.dot(action[0].new_direction, action[1].new_direction), -1.0)
        self.assertEqual(force_far_right_coll, 0.0)

    def test_homing_force(self):
        self.force_model.update_force_params(K_a=0, K_r=0, K_h=0, K_f=1, K_p=0)
        self.force_model.update_force_params(K_h=1)

        home_coll = Colloid(
            pos=np.array([500.0, 500.0, 0]),
            director=np.array([1.0, 0.0, 0.0]),
            id=1,
            velocity=np.array([10.0, 0.0, 0.0]),
            type=0,
        )

        other_coll = Colloid(
            pos=np.array([510.0, 500.0, 0.0]),
            director=np.array([1.0, 0.0, 0.0]),
            id=2,
            velocity=np.array([0.0, 10.0, 0.0]),
            type=0,
        )
        action = self.force_model.calc_action([home_coll, other_coll])
        force_home_coll = action[0].force
        force_other_coll = action[1].force
        new_dir_other_coll = action[1].new_direction
        vec_to_other_coll = self.force_model.home_pos - other_coll.pos

        new_dir_other_coll /= np.linalg.norm(new_dir_other_coll)
        vec_to_other_coll /= np.linalg.norm(vec_to_other_coll)
        dot_prod = np.dot(new_dir_other_coll, vec_to_other_coll)

        self.assertEqual(force_home_coll, 0)
        self.assertGreater(force_other_coll, 0)
        self.assertAlmostEqual(dot_prod, 1)

    def test_friction_force(self):
        self.force_model.update_force_params(K_a=0, K_r=0, K_h=0, K_f=1, K_p=0)

        coll = Colloid(
            pos=np.array([500.0, 500.0, 0]),
            director=np.array([1.0, 0.0, 0.0]),
            id=1,
            velocity=np.array([30.0, 0.0, 0.0]),
            type=0,
        )
        action = self.force_model.calc_action([coll])
        force_coll = action[0].force
        self.assertGreater(force_coll, 0)
        self.assertEqual(
            np.dot(action[0].new_direction, np.array([1.0, 0.0, 0.0])), -1.0
        )


class TestUtils(ut.TestCase):
    def test_coll_in_vision(self):
        test_coll = Colloid(pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]), id=1)
        front_coll = Colloid(
            pos=np.array([1.1, 0, 0]), director=np.array([1, 0, 0]), id=2
        )
        front_far_coll = Colloid(
            pos=np.array([100, 0, 0]), director=np.array([1, 0, 0]), id=3
        )
        slight_offset_coll = Colloid(
            pos=np.array([1, 0, 0.1]), director=np.array([1, 0, 0]), id=5
        )

        colls_in_range = swarmrl.agents.lymburn_model.get_colloids_in_vision(
            test_coll,
            [front_coll, front_far_coll, slight_offset_coll],
            vision_radius=10,
        )
        colls_shouldbe = [front_coll, slight_offset_coll]

        self.assertEqual(len(colls_in_range), len(colls_shouldbe))
        for coll in colls_shouldbe:
            self.assertIn(coll, colls_in_range)

    def test_pred_in_vision(self):
        pass


if __name__ == "__main__":
    ut.main()
