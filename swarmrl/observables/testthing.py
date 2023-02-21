"""
Unit test for the subdivided vision cones.
"""

import numpy as np

from swarmrl.observables.subdivided_vision_cones import SubdividedVisionCones

vision_class = SubdividedVisionCones(
    vision_range=10, vision_half_angle=np.pi / 2, n_cones=3, radii=[1, 2, 3, 4, 1]
)


class Colloid:
    def __init__(self, position, director, colloid_type, colloid_id):
        self.pos = position
        self.director = director
        self.type = colloid_type
        self.id = colloid_id


col0 = Colloid(np.array([0, 0, 0]), np.array([0, 1.0, 0]), 0, 0)
col1 = Colloid(np.array([0, 5, 0]), np.array([1.0, 0, 0]), 0, 1)
col2 = Colloid(np.array([0, 8, 0]), np.array([1.0, 0, 0]), 1, 2)
col3 = Colloid(np.array([-7, 8, 0]), np.array([0.0, 1.0, 0]), 1, 3)
col4 = Colloid(np.array([1, 1, 0]), np.array([0.0, 1.0, 0]), 0, 4)
colloids = [col0, col1, col2, col3, col4]
observable = vision_class.compute_observable(col0, colloids)
# assert observable[0, 0] == 1.0
# assert observable[1, 0] == 0.8
# assert observable[2, 0] == 0.0
# assert observable[0, 1] == 0.0
# assert observable[1, 1] == 0.75
# assert observable[2, 1] == 0.0
