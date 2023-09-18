"""
Test the Gumbel distribution.
"""
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.object_movement.drug_delivery import DrugDelivery, DrugTransport
from swarmrl.utils.utils import create_colloids


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

        def decay_fn(x: float):
            """
            Scaling function for the test

            Parameters
            ----------
            x : float
                    Input value.
            """
            return 1 - x

        cls.task = DrugDelivery(
            destination=np.array([800, 800, 0.0]),
            decay_fn=decay_fn,
            box_length=np.array([1000.0, 1000.0, 1000.0]),
            drug_type=1,
            particle_type=0,
            scale_factor=1000,
        )

        colloids = create_colloids(
            n_cols=10,
            type_=0,
            face_middle=True,
        )

        drug = create_colloids(n_cols=1, type_=1, dist=0)

        cls.colloids = colloids + drug

    def test_init(self):
        self.task.initialize(colloids=self.colloids)

        assert_array_equal(self.task.destination, np.array([0.8, 0.8, 0.0]))
        assert_array_equal(self.task.box_length, np.array([1000.0, 1000.0, 1000.0]))
        assert self.task.decay_fn(1) == 0

        assert_array_equal(self.task.old_drug_pos, self.colloids[-1].pos / 1000)

    def test_call(self):
        self.task.initialize(self.colloids)

        new_colloids = create_colloids(
            n_cols=10,
            type_=0,
            face_middle=True,
        )

        new_drug = create_colloids(n_cols=1, type_=1, dist=0)

        self.colloids = new_colloids + new_drug

        reward = self.task(self.colloids)

        assert np.shape(reward) == (10,)

    def test_call_drug_delivery(self):
        self.task.initialize(self.colloids)

        dist = np.linalg.norm(self.colloids[-1].pos / 1000 - self.task.destination)
        print(dist * 1000)

        while dist * 1000 > 5:
            new_drug = move_drug_to_dest(
                self.colloids[-1], 1000 * self.task.destination, delta=1, noise=0.1
            )
            self.colloids[-1] = new_drug
            reward = self.task(self.colloids)
            assert np.shape(reward) == (10,)
            # check if all arrays get a positive reward
            print(reward)
            assert np.all(reward > 0)
            dist = np.linalg.norm(self.colloids[-1].pos / 1000 - self.task.destination)


class TestDrugTransport:
    """
    Test suite for the run and tumble task.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """

        def decay_fn(x: float):
            """
            Scaling function for the test

            Parameters
            ----------
            x : float
                    Input value.
            """
            return 1 - x

        cls.task = DrugTransport(
            destination=np.array([800, 800, 0.0]),
            decay_fn=decay_fn,
            box_length=np.array([1000.0, 1000.0, 1000.0]),
            drug_type=1,
            particle_type=0,
            scale_factor=1000,
        )

        cls.colloid = Colloid(
            pos=np.array([350, 350, 0.0]),
            director=np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            id=0,
            type=0,
        )

        cls.drug = Colloid(
            pos=np.array([500, 500, 0.0]),
            director=np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            id=1,
            type=1,
        )

    def test_init(self):
        self.task.initialize(colloids=[self.colloid, self.drug])
        assert_array_equal(self.task.destination, np.array([0.5, 0.5, 0.0]))
        assert_array_equal(self.task.box_length, np.array([1000.0, 1000.0, 1000.0]))
        npt.assert_array_equal(
            self.task.historical_positions["transporter"][0],
            np.array([0.35, 0.35, 0.0]),
        )
        npt.assert_array_equal(
            self.task.historical_positions["drug"][0], np.array([0.4, 0.4, 0.0])
        )

    def test_call(self):
        rewards = []
        positions = []
        self.task.initialize(colloids=[self.colloid, self.drug])
        delta_dist = np.linalg.norm(self.colloid.pos - self.drug.pos)
        drug_dist = np.linalg.norm(self.drug.pos - self.task.destination * 1000)

        while delta_dist > 5:
            self.colloid = move_col_to_drug(self.colloid, self.drug, delta=1)
            self.drug = move_drug_to_dest(
                self.drug, self.task.destination * 1000, delta=0
            )
            delta_dist = np.linalg.norm(self.colloid.pos - self.drug.pos)
            reward = self.task([self.colloid, self.drug])
            rewards.append(reward)
            positions.append([self.colloid.pos, self.drug.pos])

        for k in range(20):
            self.colloid = move_col_to_drug(self.colloid, self.drug, delta=1)
            self.drug = move_drug_to_dest(
                self.drug, self.task.destination * 1000, delta=0
            )
            reward = self.task([self.colloid, self.drug])
            rewards.append(reward)
            positions.append([self.colloid.pos, self.drug.pos])

        while drug_dist > 5:
            if delta_dist > 5:
                self.colloid = move_col_to_drug(self.colloid, self.drug, delta=1)
                self.drug = move_drug_to_dest(
                    self.drug, self.task.destination * 1000, delta=0
                )
            else:
                self.drug = move_drug_to_dest(
                    self.drug, self.task.destination * 1000, delta=1
                )
                self.colloid = move_col_to_drug(self.colloid, self.drug, delta=2)

            drug_dist = np.linalg.norm(self.drug.pos - self.task.destination * 1000)
            reward = self.task([self.colloid, self.drug])
            rewards.append(reward)

            positions.append([self.colloid.pos, self.drug.pos])

        np.save("positions.npy", positions, allow_pickle=True)
        np.save("rewards.npy", rewards, allow_pickle=True)
