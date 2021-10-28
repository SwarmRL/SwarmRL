from EspressoMD import espresso_md
import pint
import numpy as np


# example force rules **put ML here**
class MLModel:
    def train(self, data):
        print('working really hard, I promise')


class HarmonicTrap(MLModel):
    def __init__(self, center, stiffness):
        self.center = np.asarray(center)
        self.stiffness = stiffness

    def calc_force(self, colloid):
        pos = colloid.pos_folded
        dist = pos - self.center

        return -self.stiffness * dist


class ActiveParticle(MLModel):
    def __init__(self, swim_force):
        self.swim_force = swim_force

    def calc_force(self, colloid):
        direc = colloid.director
        return self.swim_force * direc / np.linalg.norm(direc)


class ToCenterMass(MLModel):
    def __init__(self, force, only_in_front=False):
        self.force = force
        self.only_in_front = only_in_front

    def calc_force(self, colloid, other_particles):
        if self.only_in_front:
            particles_in_vision = list()
            my_pos = np.array(colloid.pos)
            my_director = colloid.director
            for other_p in other_particles:
                dist = other_p.pos - my_pos
                if np.dot(dist, my_director) > 0:
                    particles_in_vision.append(other_p)
        else:
            particles_in_vision = other_particles

        if len(particles_in_vision) == 0:
            return 3 * [0]

        center_of_mass = np.sum(np.array([p.pos * p.mass for p in particles_in_vision]), axis=0) / np.sum(
            [p.mass for p in particles_in_vision])
        dist = colloid.pos - center_of_mass

        return - self.force * dist / np.linalg.norm(dist)


com_force_rule = ToCenterMass(1000, only_in_front=True)

ureg = pint.UnitRegistry()
params = espresso_md.MDParams(n_colloids=10,
                              ureg=ureg,
                              colloid_radius=ureg.Quantity(1, 'micrometer'),
                              fluid_dyn_viscosity=ureg.Quantity(8.9, 'pascal * second'),
                              WCA_epsilon=ureg.Quantity(1e-20, 'joule'),
                              colloid_density=ureg.Quantity(2.65, 'gram / centimeter**3'),
                              temperature=ureg.Quantity(300, 'kelvin'),
                              box_length=ureg.Quantity(100, 'micrometer'),
                              time_step=ureg.Quantity(0.05, 'second'),
                              time_slice=ureg.Quantity(0.1, 'second'),
                              write_interval=ureg.Quantity(0.5, 'second'))

output_folder = './test_sim/'

system_runner = espresso_md.EspressoMD(params, seed=42, out_folder=output_folder, write_chunk_size=1000)
system_runner.setup_simulation()

for i in range(10):
    system_runner.integrate_system(500, com_force_rule)
    data_for_ML_trainer = system_runner.get_particle_data()
    com_force_rule.train(data_for_ML_trainer)
