"""
Test suite for the species search.
"""
import numpy as np
import numpy.testing as npt

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.sheep_shepp import SheepShepp


def build_circle_cols(n_colls, dist=300):
    cols = []
    pos_0 = 1000 * np.random.random(3)
    pos_0[-1] = 0
    direction_0 = np.random.random(3)
    direction_0[-1] = 0
    for i in range(n_colls - 1):
        theta = np.random.random(1)[0] * 2 * np.pi
        position = pos_0 + dist * np.array([np.cos(theta), np.sin(theta), 0])
        direction = np.random.random(3)
        direction[-1] = 0
        direction = direction / np.linalg.norm(direction)
        cols.append(Colloid(pos=position, director=direction, type=0, id=i))
    return cols


class TestSpeciesSearch:
    """
    Test suite for the gradient sensing observable.
    """

    @classmethod
    def setup_class(cls):
        shepp = Colloid(
            pos=np.array([750.0, 750.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=0,
        )
        shepp2 = Colloid(
            pos=np.array([760.0, 730.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=1,
        )
        sheep = Colloid(
            pos=np.array([780.0, 780.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=2,
        )
        sheep2 = Colloid(
            pos=np.array([790.0, 760.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=3,
        )
        center = Colloid(
            pos=np.array((500.0, 500.0, 500.0)),
            director=np.array((0.0, 1.0, 0)),
            type=2,
            id=4,
        )

        cls.colloids = [shepp, sheep, shepp2, sheep2, center]

        cls.task = SheepShepp()

    def test_init(self):
        """
        Test if the observable is initialized correctly.
        """
        self.task.initialize(colloids=self.colloids)

        # Test the concentration field initialization.
        npt.assert_array_equal(
            self.task.old_positions["sheep_pos"],
            np.array([[780.0, 780.0, 0.0], [790.0, 760.0, 0.0]]),
        )
        npt.assert_array_equal(
            self.task.old_positions["shepp_pos"],
            np.array([[750.0, 750.0, 0.0], [760.0, 730.0, 0.0]]),
        )

    def test_call(self):
        self.task.initialize(colloids=self.colloids)
        old_pos = self.task.old_positions
        reward = self.task(colloids=self.colloids)
        new_pos = self.task.old_positions
        npt.assert_equal(reward, 0.0)
        npt.assert_array_equal(old_pos["sheep_pos"], new_pos["sheep_pos"])
        npt.assert_array_equal(old_pos["shepp_pos"], new_pos["shepp_pos"])

    def test_call2(self):
        # move shepps closer to center of mass of sheep
        shepp1 = Colloid(
            pos=np.array([750.0, 750.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=0,
        )
        shepp2 = Colloid(
            pos=np.array([760.0, 730.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=1,
        )
        sheep1 = Colloid(
            pos=np.array([780.0, 780.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=2,
        )
        sheep2 = Colloid(
            pos=np.array([790.0, 760.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=3,
        )
        center = Colloid(
            pos=np.array((500.0, 500.0, 500.0)),
            director=np.array((0.0, 1.0, 0)),
            type=2,
            id=4,
        )
        colloids = [shepp1, sheep1, shepp2, sheep2, center]

        self.task.initialize(colloids=colloids)

        sheep_center_of_mass = np.array([785.0, 770.0, 0.0])
        direction1 = sheep_center_of_mass - shepp1.pos
        direction2 = sheep_center_of_mass - shepp2.pos
        direction1 /= np.linalg.norm(direction1)
        direction2 /= np.linalg.norm(direction2)
        # move shepps closer to center of mass of sheep
        new_shepp1_pos = shepp1.pos + direction1 * 10
        new_shepp2_pos = shepp2.pos + direction2 * 10
        shepp1 = Colloid(
            pos=new_shepp1_pos, director=np.array([0.0, 1.0, 0]), type=0, id=0
        )
        shepp2 = Colloid(
            pos=new_shepp2_pos, director=np.array([0.0, 1.0, 0]), type=0, id=1
        )
        moved_colloids = [shepp1, sheep1, shepp2, sheep2, center]

        reward = self.task(colloids=moved_colloids)
        assert reward > 0.0

    def test_call3(self):
        # move shepps closer to center of mass of sheep
        shepp1 = Colloid(
            pos=np.array([750.0, 750.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=0,
        )
        shepp2 = Colloid(
            pos=np.array([760.0, 730.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=1,
        )
        sheep1 = Colloid(
            pos=np.array([510.0, 500.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=2,
        )
        sheep2 = Colloid(
            pos=np.array([790.0, 760.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=3,
        )
        center = Colloid(
            pos=np.array((500.0, 500.0, 500.0)),
            director=np.array((0.0, 1.0, 0)),
            type=2,
            id=4,
        )
        colloids = [shepp1, sheep1, shepp2, sheep2, center]

        self.task.initialize(colloids=colloids)
        reward = self.task(colloids=colloids)
        assert reward == 100.0

    def test_call4(self):
        shepp1 = Colloid(
            pos=np.array([750.0, 750.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=0,
        )
        shepp2 = Colloid(
            pos=np.array([760.0, 730.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=0,
            id=1,
        )
        sheep1 = Colloid(
            pos=np.array([510.0, 500.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=2,
        )
        sheep2 = Colloid(
            pos=np.array([520.0, 460.0, 0.0]),
            director=np.array([0.0, 1.0, 0]),
            type=1,
            id=3,
        )
        center = Colloid(
            pos=np.array((500.0, 500.0, 500.0)),
            director=np.array((0.0, 1.0, 0)),
            type=2,
            id=4,
        )
        colloids = [shepp1, sheep1, shepp2, sheep2, center]
        self.task.initialize(colloids=colloids)
        reward = self.task(colloids=colloids)
        assert reward == 200.0
