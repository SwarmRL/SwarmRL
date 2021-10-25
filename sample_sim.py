from espressomd import System, visualization
import threading
import numpy as np
import tqdm
import pint

"""
Force rules calculate the force on the particle based on
* the particle properties itself (position, direction, ...)
* system properties (other particles positions)
"""


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
          'WCA_epsilon': ureg.Quantity(1e-20, 'joule'),  # fixme
          'colloid_density': ureg.Quantity(2.65, 'gram / centimeter**3'),
          'temperature': ureg.Quantity(300, 'kelvin'),
          'sim_duration': ureg.Quantity(1, 'hour'),
          'box_length': ureg.Quantity(100, 'micrometer'),
          'time_step': ureg.Quantity(0.05, 'second'),
          'time_slice': ureg.Quantity(0.1, 'second'),
          'visualize': True,
          'seed': 42
          }

# define simulation units
ureg.define('sim_length = 1e-6 meter')
ureg.define('sim_time = 1 second')
ureg.define(f"sim_energy = {params['temperature']} * boltzmann_constant")

ureg.define('sim_mass = sim_energy * sim_time**2 / sim_length**2')
ureg.define('sim_dyn_viscosity = sim_mass / (sim_length * sim_time)')

# parameter unit conversion
time_step = params['time_step'].m_as('sim_time')
# time slice: the amount of time the integrator runs before we look at the configuration and change forces
time_slice = params['time_slice'].m_as('sim_time')
box_l = np.array(3 * [params['box_length'].m_as('sim_length')])
sim_duration = params['sim_duration'].m_as('sim_time')

# system setup. Skin is a verlet list parameter that has to be set, but only affects performance
system = System(box_l=box_l)
system.time_step = time_step
system.cell_system.skin = 0.4
particle_type = 0

colloid_radius = params['colloid_radius'].m_as('sim_length')
system.non_bonded_inter[particle_type, particle_type].wca.set_params(sigma=(2 * colloid_radius) ** (-1 / 6),
                                                                     epsilon=params['WCA_epsilon'].m_as('sim_energy'))

# set up the particles. The handles will later be used to calculate/set forces
colloids = list()
colloid_mass = (4. / 3. * np.pi * params['colloid_radius'] ** 3 * params['colloid_density']).m_as('sim_mass')
for _ in range(params['n_colloids']):
    start_pos = box_l * np.random.random((3,))
    colloid = system.part.add(pos=start_pos,
                              mass=colloid_mass,
                              rinertia=3*[2. / 5. * colloid_mass * colloid_radius ** 2],
                              rotation=3 * [True])
    colloids.append(colloid)

colloid_friction_translation = 6 * np.pi * params['fluid_dyn_viscosity'].m_as('sim_dyn_viscosity') * colloid_radius
colloid_friction_rotation = 8 * np.pi * params['fluid_dyn_viscosity'].m_as('sim_dyn_viscosity') * colloid_radius ** 3

# remove overlap
system.integrator.set_steepest_descent(
    f_max=0., gamma=colloid_friction_translation, max_displacement=0.1)
system.integrator.run(1000)
system.integrator.set_vv()

# set the brownian thermostat
kT = (params['temperature'] * ureg.boltzmann_constant).m_as('sim_energy')
system.thermostat.set_brownian(kT=kT,
                               gamma=colloid_friction_translation,
                               gamma_rotation=colloid_friction_rotation,
                               seed=42)
system.integrator.set_brownian_dynamics()

# visualization is optional
visualizer = None
if params['visualize']:
    visualizer = visualization.openGLLive(system)

steps_per_slice = int(round(time_slice / time_step))
if abs(steps_per_slice - time_slice / time_step) > 1e-10:
    raise ValueError('inconsistent parameters: time_slice must be integer multiple of time_step')
n_slices = int(np.ceil(sim_duration / time_slice))


def integrate():
    for _ in tqdm.tqdm(range(n_slices)):
        system.integrator.run(steps_per_slice)
        for coll in colloids:
            force_on_colloid = com_force_rule.calc_force(coll, [c for c in colloids if c is not coll])
            coll.ext_force = force_on_colloid

        if visualizer is not None:
            visualizer.update()


if params['visualize']:
    t = threading.Thread(target=integrate)
    t.daemon = True
    t.start()
    visualizer.start()
else:
    integrate()
