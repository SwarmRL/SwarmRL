import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.searching.form_group import FromGroup


def build_cols(collist):
    """
    Helper function that builds a list of colloids from a list.

    Parameters
    ----------
    collist : list
        List of the number of colloids of each type.

    Returns
    -------
    cols : list

    """
    cols = []
    i = 0
    for type_cols, num_cols in enumerate(collist):
        for _ in range(num_cols):
            position = 1000 * onp.random.random(3)
            position[-1] = 0
            direction = onp.random.random(3)
            direction[-1] = 0
            direction = direction / onp.linalg.norm(direction)
            cols.append(Colloid(pos=position, director=direction, type=type_cols, id=i))
            i += 1
    return cols


class TestFormGroup:
    def test_init(self):
        task = FromGroup(
            box_length=np.array([1000, 1000, 1000.0]), reward_scale_factor=1000
        )
        assert task.box_length == np.array([1000, 1000, 1000.0])

    def test_initialize(self):
        colloids = build_cols([5])
        task = FromGroup(box_length=np.array([1000, 1000, 1000.0]))
        task.initialize(colloids)
        for key in task.historic_distances:
            assert key in [0, 1, 2, 3, 4]
            assert key not in [5, 6, 7, 8, 9]
            assert task.historic_distances[key] > 0

    def test_call(self):
        colloids = build_cols([5])
        task = FromGroup(
            box_length=np.array([1000, 1000, 1000.0]), reward_scale_factor=1000
        )
        task.initialize(colloids)

        # move each colloid 2 along its director
        colloids = colloids
        for i, col in enumerate(colloids):
            pos = col.pos + 2 * col.director
            colloids[i] = Colloid(
                pos=pos, director=col.director, type=col.type, id=col.id
            )

        reward = task(colloids)
        assert len(reward) == 5
        print(type(reward))
