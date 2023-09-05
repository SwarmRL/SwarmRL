"""
Test the Gumbel distribution.
"""
import numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.searching.species_avoid import SpeciesAvoid


def move_col_to_drug(colloid: Colloid, drug: Colloid, delta=1, noise=0.1):
    direction = drug.pos - colloid.pos
    direction /= np.linalg.norm(direction)
    noise_vec = noise * np.random.normal(0, 3, 3)
    noise_vec[2] = 0
    new_pos = colloid.pos + direction * delta + noise_vec
    return Colloid(
        pos=new_pos, director=colloid.director, type=colloid.type, id=colloid.id
    )


def move_drug_to_dest(drug: Colloid, destination: np.ndarray, delta=1, noise=0.1):
    direction = destination - drug.pos
    direction /= np.linalg.norm(direction)
    noise_vec = noise * np.random.normal(0, 3, 3)
    noise_vec[2] = 0
    new_pos = drug.pos + direction * delta + noise_vec
    return Colloid(pos=new_pos, director=drug.director, type=drug.type, id=drug.id)


class TestDrugDelivery:
    """
    Test suite for the run and tumble task.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """

        cls.task = SpeciesAvoid()

        prey = Colloid(
            pos=np.array([500, 500, 0]), director=np.array([1, 0, 0]), type=1, id=1
        )
        preditor = Colloid(
            pos=np.array([530, 530, 0]), director=np.array([1, 0, 0]), type=0, id=3
        )

        cls.colloids = [prey, preditor]

    def test_call(self):
        reward = self.task(self.colloids)

        assert np.shape(reward) == (1,)
        assert reward[0] == -1

        prey2 = Colloid(
            pos=np.array([100, 100, 0]), director=np.array([1, 0, 0]), type=1, id=2
        )

        self.colloids.append(prey2)

        reward = self.task(self.colloids)

        assert np.shape(reward) == (2,)
        assert reward[0] == -1
        assert reward[1] == 0

        preditor2 = Colloid(
            pos=np.array([520, 520, 0]), director=np.array([1, 0, 0]), type=0, id=4
        )

        self.colloids.append(preditor2)

        reward = self.task(self.colloids)

        assert np.shape(reward) == (2,)
        assert reward[0] == -1
        assert reward[1] == 0
