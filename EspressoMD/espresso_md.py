from espressomd import System, visualization
import numpy as np
import tqdm
import h5py
import os


class EspressoMD:
    def __init__(self, md_params, out_folder='.'):
        self.params = md_params
        self.out_folder = out_folder
        self.system = System(box_l=3 * [1])
        self.colloids = list()

    def setup_simulation(self):
        ureg = self.params['ureg']
        # define simulation units
        ureg.define('sim_length = 1e-6 meter')
        ureg.define('sim_time = 1 second')
        ureg.define(f"sim_energy = {self.params['temperature']} * boltzmann_constant")

        ureg.define('sim_mass = sim_energy * sim_time**2 / sim_length**2')
        ureg.define('sim_dyn_viscosity = sim_mass / (sim_length * sim_time)')

        # parameter unit conversion
        time_step = self.params['time_step'].m_as('sim_time')
        # time slice: the amount of time the integrator runs before we look at the configuration and change forces
        time_slice = self.params['time_slice'].m_as('sim_time')
        box_l = np.array(3 * [self.params['box_length'].m_as('sim_length')])

        # system setup. Skin is a verlet list parameter that has to be set, but only affects performance
        self.system.box_l = box_l
        self.system.time_step = time_step
        self.system.cell_system.skin = 0.4
        particle_type = 0

        colloid_radius = self.params['colloid_radius'].m_as('sim_length')
        self.system.non_bonded_inter[particle_type, particle_type].wca.set_params(
            sigma=(2 * colloid_radius) ** (-1 / 6),
            epsilon=self.params['WCA_epsilon'].m_as(
                'sim_energy'))

        # set up the particles. The handles will later be used to calculate/set forces

        colloid_mass = (4. / 3. * np.pi * self.params['colloid_radius'] ** 3 * self.params['colloid_density']).m_as(
            'sim_mass')
        for _ in range(self.params['n_colloids']):
            start_pos = box_l * np.random.random((3,))
            colloid = self.system.part.add(pos=start_pos,
                                           mass=colloid_mass,
                                           rinertia=3 * [2. / 5. * colloid_mass * colloid_radius ** 2],
                                           rotation=3 * [True])
            self.colloids.append(colloid)

        colloid_friction_translation = 6 * np.pi * self.params['fluid_dyn_viscosity'].m_as(
            'sim_dyn_viscosity') * colloid_radius
        colloid_friction_rotation = 8 * np.pi * self.params['fluid_dyn_viscosity'].m_as(
            'sim_dyn_viscosity') * colloid_radius ** 3

        # remove overlap
        self.system.integrator.set_steepest_descent(
            f_max=0., gamma=colloid_friction_translation, max_displacement=0.1)
        self.system.integrator.run(1000)

        # set the brownian thermostat
        kT = (self.params['temperature'] * ureg.boltzmann_constant).m_as('sim_energy')
        self.system.thermostat.set_brownian(kT=kT,
                                            gamma=colloid_friction_translation,
                                            gamma_rotation=colloid_friction_rotation,
                                            seed=42)
        self.system.integrator.set_brownian_dynamics()

        # set integrator params
        steps_per_slice = int(round(time_slice / time_step))
        self.params.update({'steps_per_slice': steps_per_slice})
        if abs(steps_per_slice - time_slice / time_step) > 1e-10:
            raise ValueError('inconsistent parameters: time_slice must be integer multiple of time_step')

        self._init_h5_output()

    def _init_h5_output(self):
        self.h5_filename = self.out_folder + '/trajectory.hdf5'
        os.makedirs(self.out_folder, exist_ok=True)
        self.write_idx = 0.

        self.traj_holder = {'Times': list(),
                            'Unwrapped_Positions': list(),
                            'Velocities': list(),
                            'Directors': list()}

        with h5py.File(self.h5_filename, 'a') as h5_outfile:
            part_group = h5_outfile.require_group('colloids')
            dataset_kwargs = dict(compression="gzip")

            part_group.require_dataset('Velocities',
                                       shape=(self.params['n_colloids'], int(1e8), 3),
                                       dtype=float,
                                       **dataset_kwargs)
        self.h5_time_steps_written = 0

    def _write_chunk_to_file(self):
        values = np.stack(self.traj_holder['Velocities'])
        n_new_timesteps = len(values)

        with h5py.File(self.h5_filename, 'a') as h5_outfile:
            part_group = h5_outfile['colloids']
            dataset = part_group['Velocities']

            # mdsuite format is (particle_id, time_step, dimension) -> swapaxes
            dataset[:, self.h5_time_steps_written:self.h5_time_steps_written+n_new_timesteps, :] = np.swapaxes(values, 0, 1)

        self.h5_time_steps_written += n_new_timesteps

        for val in self.traj_holder.values():
            val.clear()

    def integrate_system(self, n_slices, force_rule):
        for _ in tqdm.tqdm(range(n_slices)):
            if self.system.time >= self.params['write_interval'].m_as('sim_time') * self.write_idx:
                self.traj_holder['Times'].append(self.system.time)
                self.traj_holder['Unwrapped_Positions'].append(np.stack([c.pos for c in self.colloids]))
                self.traj_holder['Velocities'].append(np.stack([c.v for c in self.colloids]))
                self.traj_holder['Directors'].append(np.stack([c.director for c in self.colloids]))

                if len(self.traj_holder['Times']) >= self.params['write_chunk_size']:
                    self._write_chunk_to_file()
                    self.write_idx += 1

            self.system.integrator.run(self.params['steps_per_slice'])
            for coll in self.colloids:
                force_on_colloid = force_rule.calc_force(coll, [c for c in self.colloids if c is not coll])
                coll.ext_force = force_on_colloid

    def get_particle_data(self):
        return {'Unwrapped_Positions': np.stack([c.pos for c in self.colloids]),
                'Velocities': np.stack([c.v for c in self.colloids]),
                'Directors': np.stack([c.director for c in self.colloids])}
