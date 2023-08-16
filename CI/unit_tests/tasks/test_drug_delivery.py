"""
Test the Gumbel distribution.
"""
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.object_movement.drug_delivery import DrugDelivery, DrugTransport
from swarmrl.utils.utils import create_colloids


def move_col_to_drug(colloid: Colloid, drug: Colloid, delta=3, noise=False):
    direction = drug.pos - colloid.pos
    direction /= np.linalg.norm(direction)
    if noise:
        noise = np.random.normal(0, 3, 3)
    else:
        noise = np.zeros(3)
    new_pos = colloid.pos + direction * delta + noise
    return Colloid(
        pos=new_pos, director=colloid.director, type=colloid.type, id=colloid.id
    )


def move_drug_to_dest(drug: Colloid, destination: np.ndarray, delta=3, noise=False):
    direction = destination - drug.pos
    direction /= np.linalg.norm(direction)
    if noise:
        noise = np.random.normal(0, 3, 3)
    else:
        noise = np.zeros(3)
    new_pos = drug.pos + direction * delta + noise
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

        assert_array_equal(self.task.destination, np.array([0.5, 0.5, 0.0]))
        assert_array_equal(self.task.box_length, np.array([1000.0, 1000.0, 1000.0]))
        assert self.task.decay_fn(1) == 0

        assert self.task.colloid_indices["drug"] == [10]
        assert_array_equal(self.task.colloid_indices["transporter"], np.arange(10))

        drug_pos = (
            np.array([col.pos for col in self.colloids if col.type == 1])
            / self.task.box_length
        )
        transporter_pos = (
            np.array([col.pos for col in self.colloids if col.type == 0])
            / self.task.box_length
        )

        assert_array_equal(drug_pos, self.task.historical_positions["drug"])
        assert_array_equal(
            transporter_pos, self.task.historical_positions["transporter"]
        )

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

        extra_new_colloids = []

        for col in new_colloids:
            new_pos = col.pos + 3 * col.director
            new_colloid = Colloid(
                pos=new_pos, director=col.director, id=col.id, type=col.type
            )
            extra_new_colloids.append(new_colloid)

        self.colloids = extra_new_colloids + new_drug

        positive_reward = self.task(self.colloids)

        old_drug_pos = self.task.historical_positions["drug"]
        npt.assert_almost_equal(positive_reward, 3 * np.ones_like(positive_reward))

        drug_dest_vector = self.task.destination - self.colloids[10].pos
        drug_dest_vector /= np.linalg.norm(drug_dest_vector)
        new_new_drug = Colloid(
            pos=new_drug[0].pos - 3 * drug_dest_vector,
            director=new_drug[0].director,
            id=new_drug[0].id,
            type=new_drug[0].type,
        )

        self.colloids = extra_new_colloids + [new_new_drug]
        drug_delivery_reward = self.task(self.colloids)

        npt.assert_almost_equal(
            drug_delivery_reward, 30 * np.ones_like(drug_delivery_reward), decimal=-1
        )

        new_drug_pos = self.task.historical_positions["drug"]

        npt.assert_raises(
            AssertionError, assert_array_equal, old_drug_pos, new_drug_pos
        )


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
            pos=np.array([0, 0, 0.0]),
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
        assert_array_equal(self.task.destination, np.array([0.8, 0.8, 0.0]))
        assert_array_equal(self.task.box_length, np.array([1000.0, 1000.0, 1000.0]))
        npt.assert_array_equal(
            self.task.historical_positions["transporter"][0], np.array([0.0, 0.0, 0.0])
        )
        npt.assert_array_equal(
            self.task.historical_positions["drug"][0], np.array([0.5, 0.5, 0.0])
        )

    def test_call(self):
        rewards = []
        positions = []
        self.task.initialize(colloids=[self.colloid, self.drug])
        delta_dist = np.linalg.norm(self.colloid.pos - self.drug.pos)

        while delta_dist > 8:
            self.colloid = move_col_to_drug(self.colloid, self.drug)
            delta_dist = np.linalg.norm(self.colloid.pos - self.drug.pos)
            reward = self.task([self.colloid, self.drug])
            rewards.append(reward)
            positions.append([self.colloid.pos, self.drug.pos])

        drug_dest_dist = np.linalg.norm(self.drug.pos - self.task.destination)
        while drug_dest_dist > 8:
            self.drug = move_drug_to_dest(self.drug, self.task.destination * 1000)
            self.colloid = move_col_to_drug(self.colloid, self.drug)
            drug_dest_dist = np.linalg.norm(
                self.drug.pos - self.task.destination * 1000
            )
            reward = self.task([self.colloid, self.drug])
            rewards.append(reward)
            positions.append([self.colloid.pos, self.drug.pos])

        for i in range(20):
            reward = self.task([self.colloid, self.drug])
            rewards.append(reward)
            positions.append([self.colloid.pos, self.drug.pos])

        np.save("positions.npy", positions, allow_pickle=True)
        np.save("rewards.npy", rewards, allow_pickle=True)
