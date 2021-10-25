import simulation_runner
import pint
import numpy as np


# example force rules **put ML here**
class HarmonicTrap:
    def __init__(self, center, stiffness):
        self.center = np.asarray(center)
        self.stiffness = stiffness

    def calc_force(self, colloid):
        pos = colloid.pos_folded
        dist = pos - self.center

        return -self.stiffness * dist


class ActiveParticle:
    def __init__(self, swim_force):
        self.swim_force = swim_force

    def calc_force(self, colloid):
        direc = colloid.director
        return self.swim_force * direc / np.linalg.norm(direc)


class ToCenterMass:
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

params = {'n_colloids': 10,
          'colloid_radius': ureg.Quantity(1, 'micrometer'),
          'fluid_dyn_viscosity': ureg.Quantity(8.9, 'pascal * second'),
          'WCA_epsilon': ureg.Quantity(1e-20, 'joule'),
          'colloid_density': ureg.Quantity(2.65, 'gram / centimeter**3'),
          'temperature': ureg.Quantity(300, 'kelvin'),
          'sim_duration': ureg.Quantity(1, 'hour'),
          'box_length': ureg.Quantity(100, 'micrometer'),
          'time_step': ureg.Quantity(0.05, 'second'),
          'time_slice': ureg.Quantity(0.1, 'second'),
          'write_interval': ureg.Quantity(0.1, 'second'),
          'write_chunk_size': 1000,
          'visualize': True,
          'seed': 42,
          'ureg': ureg
          }

output_folder = './'  #'/work/clohrmann/bechinger_swimmers/test_sim'

simulation_runner.integrate_system(params, com_force_rule, out_folder=output_folder)

