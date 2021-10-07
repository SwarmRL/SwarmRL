from espressomd import System, visualization
import threading
import numpy as np
import tqdm

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
    def __init__(self, force):
        self.force = force

    def calc_force(self, colloid, other_particles):
        center_of_mass = np.sum(np.array([p.pos * p.mass for p in other_particles]), axis=0) / np.sum(
            [p.mass for p in other_particles])
        dist = colloid.pos - center_of_mass

        return - self.force * dist / np.linalg.norm(dist)


# system parameters
n_colloids = 10
colloid_radius = 1
fluid_dyn_viscosity = 0.01
WCA_epsilon = 1
colloid_mass = 0.01
kT = 0.01
time_step = 0.01
# time slice: the amount of time the integrator runs before we look at the configuration and change forces
time_slice = 0.1
box_l = np.array(3 * [100])
sim_duration = 1e5

# system setup. Skin is a verlet list parameter that has to be set, but only affects performance
system = System(box_l=box_l)
system.time_step = time_step
system.cell_system.skin = 0.4
particle_type = 0

# Langevin thermostat is one option, brownian dynamics can also be used.
# rotational degrees of freedom are integrated+thermalised as well
friction = 6*np.pi* fluid_dyn_viscosity*colloid_radius
system.thermostat.set_langevin(kT=kT, gamma=friction, gamma_rotation=friction, seed=42)

system.non_bonded_inter[particle_type, particle_type].wca.set_params(sigma=(2 * colloid_radius) ** (-1 / 6),
                                                                     epsilon=WCA_epsilon)

# here we set up the particle. The handle will later be used to calculate/set forces
colloids = list()
for _ in range(n_colloids):
    start_pos = box_l * np.random.random((3,))
    start_velocity = 0.1 * np.random.random((3,))
    colloid = system.part.add(pos=start_pos, v=start_velocity, mass=colloid_mass, rotation=3 * [True])
    colloids.append(colloid)

com_force_rule = ToCenterMass(0.2)

# visualization is optional
visualizer = visualization.openGLLive(system)

# we let espresso integrate for a few steps before we step in and change the forces
steps_per_slice = int(round(time_slice / time_step))
n_slices = int(np.ceil(sim_duration / time_slice))


def integrate():
    for _ in tqdm.tqdm(range(n_slices)):
        system.integrator.run(steps_per_slice)
        for coll in colloids:
            force_on_colloid = com_force_rule.calc_force(coll, [c for c in colloids if c is not coll])
            coll.ext_force = force_on_colloid

        if visualizer is not None:
            visualizer.update()


t = threading.Thread(target=integrate)
t.daemon = True
t.start()
visualizer.start()
