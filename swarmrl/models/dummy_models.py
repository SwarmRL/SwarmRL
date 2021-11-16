from . import interaction_model
import numpy as np
import torch


class ConstForce(interaction_model.InteractionModel):
    def __init__(self, force):
        self.force = force

    def calc_force(self, colloid, other_colloids) -> np.ndarray:
        return self.force

    def calc_torque(self, colloid, other_colloids) -> np.ndarray:
        return np.zeros(3)


class ToConstOrientation(interaction_model.InteractionModel):
    def __init__(self, orientation):
        self.orientation = orientation

    def calc_force(self, colloid, other_colloids) -> np.ndarray:
        return np.zeros((3,))

    def calc_torque(self, colloid, other_colloids) -> np.ndarray:
        return np.zeros((3,))

    def calc_new_direction(self, colloid, other_colloids) -> np.ndarray:
        return self.orientation


class ActiveParticle(interaction_model.InteractionModel):
    def __init__(self, swim_force):
        self.swim_force = swim_force

    def calc_force(self, colloid, other_colloids):
        direc = colloid.director
        return self.swim_force * direc / np.linalg.norm(direc)

    def calc_torque(self, colloid, other_colloids) -> np.ndarray:
        return np.zeros(3)

    def forward(self, colloids: torch.Tensor, state: torch.Tensor = None):
        pass


class ToCenterMass(interaction_model.InteractionModel):
    def __init__(self, force, vision_angle=2 * np.pi):
        self.force = force
        self.vision_angle = vision_angle

    def calc_force(self, colloid, other_particles):
        particles_in_vision = list()
        my_pos = np.array(colloid.pos)
        my_director = colloid.director
        for other_p in other_particles:
            dist = other_p.pos - my_pos
            dist /= np.linalg.norm(dist)
            if np.arccos(np.dot(dist, my_director)) < self.vision_angle / 2.0:
                particles_in_vision.append(other_p)

        if len(particles_in_vision) == 0:
            return 3 * [0]

        center_of_mass = np.sum(
            np.array([p.pos * p.mass for p in particles_in_vision]), axis=0
        ) / np.sum([p.mass for p in particles_in_vision])
        dist = colloid.pos - center_of_mass

        return -self.force * dist / np.linalg.norm(dist)

    def calc_torque(self, colloid, other_colloids) -> np.ndarray:
        return np.zeros(3)

    def forward(self, colloids: torch.Tensor, state: torch.Tensor = None):
        pass
