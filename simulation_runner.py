from espressomd import System, visualization
import threading
import numpy as np
import tqdm
import h5py


def integrate_system(params, force_rule, out_folder='./'):
    """
    TODO: if we want to run multiple simulations from the same script, 'system' must be an argument
    because espresso does not allow multiple instances
    """
    ureg = params['ureg']
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
                                                                         epsilon=params['WCA_epsilon'].m_as(
                                                                             'sim_energy'))

    # set up the particles. The handles will later be used to calculate/set forces
    colloids = list()
    colloid_mass = (4. / 3. * np.pi * params['colloid_radius'] ** 3 * params['colloid_density']).m_as('sim_mass')
    for _ in range(params['n_colloids']):
        start_pos = box_l * np.random.random((3,))
        colloid = system.part.add(pos=start_pos,
                                  mass=colloid_mass,
                                  rinertia=3 * [2. / 5. * colloid_mass * colloid_radius ** 2],
                                  rotation=3 * [True])
        colloids.append(colloid)

    colloid_friction_translation = 6 * np.pi * params['fluid_dyn_viscosity'].m_as('sim_dyn_viscosity') * colloid_radius
    colloid_friction_rotation = 8 * np.pi * params['fluid_dyn_viscosity'].m_as(
        'sim_dyn_viscosity') * colloid_radius ** 3

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

    import os
    os.makedirs(out_folder, exist_ok=True)
    h5_outfile = h5py.File(f'{out_folder}/trajectory.hdf5', 'w')

    def _integrate():
        traj_holder = {'time': list(),
                       'step': list(),
                       'positions_unfolded': list(),
                       'velocities': list(),
                       'directors': list()}
        write_idx = 0.

        def _write_chunk(traj_holder, h5_outfile):
            # todo the actual writing
            for val in traj_holder.values():
                val.clear()

        for _ in tqdm.tqdm(range(n_slices)):

            if system.time >= params['write_interval'].m_as('sim_time') * write_idx:
                traj_holder['time'].append(system.time)
                traj_holder['step'].append(write_idx)
                traj_holder['positions_unfolded'].append(np.stack([c.pos for c in colloids]))
                traj_holder['velocities'].append(np.stack([c.v for c in colloids]))
                traj_holder['directors'].append(np.stack([c.director for c in colloids]))

                if len(traj_holder['time']) > params['write_chunk_size']:
                    _write_chunk(traj_holder, h5_outfile)

                write_idx += 1

            system.integrator.run(steps_per_slice)
            for coll in colloids:
                force_on_colloid = force_rule.calc_force(coll, [c for c in colloids if c is not coll])
                coll.ext_force = force_on_colloid

            if visualizer is not None:
                visualizer.update()

    if params['visualize']:
        t = threading.Thread(target=_integrate)
        t.daemon = True
        t.start()
        visualizer.start()
    else:
        _integrate()
