import dataclasses
import unittest as ut
import numpy as np
import context
import swarmrl.models.bechinger_models


@dataclasses.dataclass
class MockColloid:
    id: int = 0
    pos: np.ndarray = np.array([0, 0, 0])
    director: np.ndarray = np.array([1, 0, 0])
    pos_folded: np.ndarray = np.array([0, 0, 0])
    mass: float = 1

    def __eq__(self, other):
        return self.id == other.id and \
               np.all(self.pos == other.pos) and \
               np.all(self.director == other.director) and \
               np.all(self.pos_folded == other.pos_folded) and \
               self.mass == other.mass


class TestLavergne(ut.TestCase):
    def setUp(self) -> None:
        self.act_force = 1.234
        self.vision_half_angle = np.pi / 4.
        self.perception_threshold = 0.5

        self.force_model = swarmrl.models.bechinger_models.Lavergne2019(vision_half_angle=self.vision_half_angle,
                                                                        act_force=self.act_force,
                                                                        perception_threshold=self.perception_threshold)

    def test_force(self):
        orientation = np.array([1, 0, 0])
        test_coll = MockColloid(pos=[0, 0, 0], director=orientation)
        coll_front = MockColloid(pos=[100, 0, 0])
        coll_back = MockColloid(pos=[-0.01, 0, 0])
        coll_side = MockColloid(pos=[0, 0.01, 0])

        force = self.force_model.calc_force(test_coll, [coll_front, coll_back, coll_side])
        # front colloid too far, back not visible
        np.testing.assert_array_almost_equal(force, np.zeros((3,)))

        coll_front.pos = [0.1, 0, 0]
        force = self.force_model.calc_force(test_coll, [coll_front, coll_back, coll_side])
        # front close -> activity along orientation
        np.testing.assert_array_almost_equal(force, self.act_force * orientation)


class TestBaeuerle(ut.TestCase):
    def setUp(self) -> None:
        self.act_force = 1.234
        self.vision_half_angle = np.pi / 4.
        self.detection_radius_position = 1.1
        self.detection_radius_orientation = 0.5
        self.angular_deviation = np.pi / 8.

        self.force_model = swarmrl.models.bechinger_models.Baeuerle2020(act_force=self.act_force,
                                                                        detection_radius_orientation=self.detection_radius_orientation,
                                                                        detection_radius_position=self.detection_radius_position,
                                                                        vision_half_angle=self.vision_half_angle,
                                                                        angular_deviation=self.angular_deviation)

    def test_force(self):
        test_coll = MockColloid(pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]))
        front_coll = MockColloid(pos=np.array([1, 0.1, 0]), director=[0, 1, 0])
        front_close_coll = MockColloid(pos=[0.2, 0.1, 0], director=[0, -1, 0])
        front_far_coll = MockColloid(pos=[10, 0, 0], director=[0, 1, 0])
        side_coll = MockColloid(pos=[0, 0.1, 0])

        force = self.force_model.calc_force(test_coll, [front_coll, front_close_coll, front_far_coll, side_coll])
        force_norm = np.linalg.norm(force)
        self.assertAlmostEqual(force_norm, self.act_force)
        # com determied by front_coll and front_close_coll
        # orientation determined by front_close_coll
        com = (front_coll.pos + front_close_coll.pos)/2.
        com_direction = com / np.linalg.norm(com)

        # force must be to the right of the com (coll_front_close point to -y)
        self.assertAlmostEqual(np.arccos(np.dot(com_direction, force / force_norm)), self.angular_deviation)
        self.assertGreater(com[1], force[1])


class TestUtils(ut.TestCase):
    def test_coll_in_vision(self):
        test_coll = MockColloid(pos=np.array([0, 0, 0]), director=np.array([1, 0, 0]))
        front_coll = MockColloid(pos=np.array([1.1, 0, 0]))
        front_far_coll = MockColloid(pos=np.array([100, 0, 0]))
        side_coll = MockColloid(pos=np.array([0, 0.2, 0]))
        slight_offset_coll = MockColloid(pos=np.array([1, 0, 0.1]))

        colls_in_range = swarmrl.models.bechinger_models.get_colloids_in_vision(test_coll,
                                                                                [front_coll, front_far_coll, side_coll,
                                                                                 slight_offset_coll],
                                                                                vision_half_angle=np.pi / 4.,
                                                                                vision_range=10)
        colls_shouldbe = [front_coll, slight_offset_coll]

        self.assertEqual(len(colls_in_range), len(colls_shouldbe))
        for coll in colls_shouldbe:
            self.assertIn(coll, colls_in_range)


if __name__ == '__main__':
    ut.main()
