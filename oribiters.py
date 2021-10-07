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
class HarmonicTrap():
    def __init__(self, center, stiffness):
        self.center = np.asarray(center)
        self.stiffness = stiffness

    def calc_force(self, colloid):
        pos = colloid.pos_folded
        dist = pos - self.center

        return -self.stiffness * dist


class RandomAroundDirector():
    def __init__(self, force_norm):
        self.force_norm = force_norm

    def calc_force(self, colloid):
        direc = colloid.director
        rnd_force = direc + 0.1 * np.random.random((3,))
        return self.force_norm * rnd_force / np.linalg.norm(rnd_force)


# system parameters
colloid_mass = 0.01
kT = 0.01
friction = 1 # 6 pi mu r
time_step = 0.0001
# time slice: the amount of time the integrator runs before we look at the configuration and change forces
time_slice = 0.1
box_l = np.array(3 * [100])
sim_duration = 1e12

# system setup. Skin is a verlet list parameter that has to be set, but only affects performance
system = System(box_l=box_l)
system.time_step = time_step
system.cell_system.skin = 0.4

# Langevin thermostat is one option, brownian dynamics can also be used.
# rotational degrees of freedom are integrated+thermalised as well
system.thermostat.set_langevin(kT=kT, gamma=friction, gamma_rotation=friction, seed=42)

center = 0.5 * box_l
start_pos = center + np.array([0.3 * box_l[0], 0, 0])
start_velocity = [0, 10, 0]

# here we set up the particle. The handle will later be used to calculate/set forces
colloid = system.part.add(pos=start_pos, v=start_velocity, mass=colloid_mass, rotation=3 * [True])
my_force_rule = RandomAroundDirector(0.2)

# visualization is optional
visualizer = visualization.openGLLive(system)

# we let espresso integrate for a few steps before we step in and change the forces
steps_per_slice = int(round(time_slice / time_step))
n_slices = int(np.ceil(sim_duration / time_slice))


def integrate():
    for _ in tqdm.tqdm(range(n_slices)):
        system.integrator.run(steps_per_slice)
        colloid.ext_force = my_force_rule.calc_force(colloid)

        if visualizer is not None:
            visualizer.update()


t = threading.Thread(target=integrate)
t.daemon = True
t.start()
visualizer.start()
