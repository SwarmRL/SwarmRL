"""
Test the SwarmRL agents
"""
import jax.numpy as np
from numpy.testing import assert_array_equal

from swarmrl.agents.colloid import Colloid
from swarmrl.agents.swarm import create_swarm


class TestAgents:
    """
    Test suite for agents.
    """

    def test_colloid_agent(self):
        """
        Test the colloid agent.
        """
        colloid = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=0,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=0,
        )

        assert colloid.pos.shape == (3,)
        assert colloid.director.shape == (3,)
        assert colloid.velocity.shape == (3,)
        assert colloid.id == 0
        assert colloid.type == 0

    def test_colloid_unwrapping(self):
        """
        Test the pytree aspects of the colloid.
        """
        colloid = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=0,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=0,
        )

        colloid_tree = colloid.tree_flatten()
        colloid_unwrapped = Colloid.tree_unflatten(None, colloid_tree[0])

        assert colloid == colloid_unwrapped

    def test_swarm_agent(self):
        """
        Test the swarm agent.
        """
        colloid_1 = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=0,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=0,
        )

        colloid_2 = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=1,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=0,
        )

        colloid_3 = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=2,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=1,
        )

        swarm = create_swarm([colloid_1, colloid_2, colloid_3])

        assert swarm.pos.shape == (3, 3)
        assert swarm.director.shape == (3, 3)
        assert swarm.velocity.shape == (3, 3)
        assert swarm.id.shape == (3, 1)
        assert swarm.type.shape == (3, 1)

        indices = swarm.type_indices
        assert_array_equal(indices[0], [0, 1])
        assert_array_equal(indices[1], [2])

    def test_swarm_partition(self):
        """
        Test the partial swarm extraction methods.
        """
        colloid_1 = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=0,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=0,
        )

        colloid_2 = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=1,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=0,
        )

        colloid_3 = Colloid(
            pos=np.array([0.0, 0.0, 0.0]),
            director=np.array([0.0, 0.0, 1.0]),
            id=2,
            velocity=np.array([0.0, 0.0, 0.0]),
            type=1,
        )

        swarm_full = create_swarm([colloid_1, colloid_2, colloid_3])

        swarm_large = swarm_full.get_species_swarm(0)
        swarm_small = swarm_full.get_species_swarm(1)

        assert swarm_large.pos.shape == (2, 3)
        assert swarm_large.director.shape == (2, 3)
        assert swarm_large.velocity.shape == (2, 3)
        assert swarm_large.id.shape == (2, 1)
        assert swarm_large.type.shape == (2, 1)

        assert swarm_small.pos.shape == (1, 3)
        assert swarm_small.director.shape == (1, 3)
        assert swarm_small.velocity.shape == (1, 3)
        assert swarm_small.id.shape == (1, 1)
        assert swarm_small.type.shape == (1, 1)

        assert_array_equal(swarm_large.pos, swarm_full.pos[:2, :])
        assert_array_equal(swarm_large.director, swarm_full.director[:2, :])
        assert_array_equal(swarm_large.velocity, swarm_full.velocity[:2, :])
        assert_array_equal(swarm_large.id, swarm_full.id[:2, :])
        assert_array_equal(swarm_large.type, swarm_full.type[:2, :])

        assert_array_equal(swarm_small.pos, swarm_full.pos[2:, :])
        assert_array_equal(swarm_small.director, swarm_full.director[2:, :])
        assert_array_equal(swarm_small.velocity, swarm_full.velocity[2:, :])
        assert_array_equal(swarm_small.id, swarm_full.id[2:, :])
        assert_array_equal(swarm_small.type, swarm_full.type[2:, :])
