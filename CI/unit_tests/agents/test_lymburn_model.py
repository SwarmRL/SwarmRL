import unittest as ut
import swarmrl.agents.lymburn_model
import numpy as np
from swarmrl.components.colloid import Colloid


class TestLymburnModel(ut.TestCase):
    def setUp(self):
        # Set up any necessary test data or objects
        self.force_params = {"K_a": 0.1,
                             "K_r": 0.1,
                             "K_h": 0.1,
                             "K_f": 1,
                             "K_p": 100}
        self.pred_movement = swarmrl.agents.lymburn_model.harmonic_2d

        self.pred_params = [1000, 0.4, 0]
        self.force_model = swarmrl.agents.lymburn_model.Lymburn(
            force_params=self.force_params,
            pred_movement=self.pred_movement,
            pred_params=self.pred_params,
            detection_radius_position_colls=10.,
            detection_radius_position_pred=20.,
        )

    def test_force(self):
        orientation1 = np.array([1, 0, 0])
        orientation2 = np.array([0, 1, 0])
        test_coll = Colloid(pos=np.array([500, 500, 0]), director=orientation1, 
                            id=1, type=0)
        coll_front = Colloid(pos=np.array([510, 500, 0]),
                             director=orientation1, id=2, type=0)
        coll_back = Colloid(pos=np.array([505, 500, 0]), director=orientation2,
                            id=3, type=0)
        coll_side = Colloid(pos=np.array([500, 510, 0]), director=orientation2,
                            id=4, type=0)
        pred = Colloid(pos=np.array([500, 480, 0]), director=orientation2, id=5, type=1)

        colloids = [test_coll, coll_front, coll_back, coll_side, pred]

        action = self.force_model.calc_action(colloids)
        force_is = action[0].force  
        self.assertAlmostEqual(force_is, 0)


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
            vision_range=10,
        )
        colls_shouldbe = [front_coll, slight_offset_coll]

        self.assertEqual(len(colls_in_range), len(colls_shouldbe))
        for coll in colls_shouldbe:
            self.assertIn(coll, colls_in_range)


if __name__ == '__main__':
    ut.main()
