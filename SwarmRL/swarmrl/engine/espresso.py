from espressomd import System, visualization
import numpy as np
import h5py
import os
import logging
import dataclasses
import pint

from .engine import Engine
import swarmrl.models.interaction_model


@dataclasses.dataclass()
class MDParams:
    n_colloids: int
    ureg: pint.UnitRegistry
    box_length: pint.Quantity
    colloid_radius: pint.Quantity
    colloid_density: pint.Quantity
    fluid_dyn_viscosity: pint.Quantity
    WCA_epsilon: pint.Quantity
    temperature: pint.Quantity
    time_step: pint.Quantity
    time_slice: pint.Quantity
    write_interval: pint.Quantity


class EspressoMD(Engine):
    def __init__(self, md_params, n_dims=3, seed=42, out_folder='.', loglevel=logging.DEBUG, write_chunk_size=100):
        self.params = md_params
        self.out_folder = out_folder
        self.seed = seed
        self.n_dims = n_dims

        self._init_unit_system()
        self._init_h5_output(write_chunk_size)
        self._init_logger(loglevel)

        self.system = System(box_l=3 * [1])
        self.colloids = list()

    def _init_unit_system(self):
        self.ureg = self.params.ureg

        # three basis units chosen arbitrarily
        self.ureg.define('sim_length = 1e-6 meter')
        self.ureg.define('sim_time = 1 second')
        self.ureg.define(f"sim_energy = 293 kelvin * boltzmann_constant")

        # derived units
        self.ureg.define('sim_mass = sim_energy * sim_time**2 / sim_length**2')
        self.ureg.define('sim_dyn_viscosity = sim_mass / (sim_length * sim_time)')
        self.ureg.define('sim_force = sim_mass * sim_length / sim_time**2')

    def _init_h5_output(self, write_chunk_size):
        self.write_chunk_size = write_chunk_size
        self.h5_filename = self.out_folder + '/trajectory.hdf5'
        os.makedirs(self.out_folder, exist_ok=True)
        self.traj_holder = {'Times': list(),
                            'Unwrapped_Positions': list(),
                            'Velocities': list(),
                            'Directors': list()}

        with h5py.File(self.h5_filename, 'a') as h5_outfile:
            part_group = h5_outfile.require_group('colloids')
            dataset_kwargs = dict(compression="gzip")
            traj_len = 1000

            part_group.require_dataset('Times',
                                       shape=(traj_len,),
                                       dtype=float,
                                       **dataset_kwargs)
            part_group.require_dataset('Unwrapped_Positions',
                                       shape=(self.params.n_colloids, traj_len, 3),
                                       dtype=float,
                                       **dataset_kwargs)
            part_group.require_dataset('Velocities',
                                       shape=(self.params.n_colloids, traj_len, 3),
                                       dtype=float,
                                       **dataset_kwargs)
            part_group.require_dataset('Directors',
                                       shape=(self.params.n_colloids, traj_len, 3),
                                       dtype=float,
                                       **dataset_kwargs)
        self.write_idx = 0.
        self.h5_time_steps_written = 0

    def _init_logger(self, loglevel):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        formatter = logging.Formatter(fmt="[%(levelname)-10s] %(asctime)s %(message)s",
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(f'{self.out_folder}/simulation_log.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(loglevel)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(loglevel)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _write_traj_chunk_to_file(self):
        n_new_timesteps = len(self.traj_holder['Times'])

        with h5py.File(self.h5_filename, 'a') as h5_outfile:
            part_group = h5_outfile['colloids']
            for key in self.traj_holder.keys():
                dataset = part_group[key]
                values = np.stack(self.traj_holder[key])
                if key != 'Times':
                    # mdsuite format is (particle_id, time_step, dimension) -> swapaxes
                    values = np.swapaxes(values, 0, 1)
                    dataset.resize(self.h5_time_steps_written + n_new_timesteps, axis=1)
                    dataset[:, self.h5_time_steps_written:self.h5_time_steps_written + n_new_timesteps, :] = values
                else:
                    dataset.resize(self.h5_time_steps_written + n_new_timesteps, axis=0)
                    dataset[self.h5_time_steps_written:self.h5_time_steps_written + n_new_timesteps] = values

        self.logger.debug(f'wrote {n_new_timesteps} time steps to hdf5 file')
        self.h5_time_steps_written += n_new_timesteps

    def setup_simulation(self):
        # parameter unit conversion
        time_step = self.params.time_step.m_as('sim_time')
        # time slice: the amount of time the integrator runs before we look at the configuration and change forces
        time_slice = self.params.time_slice.m_as('sim_time')
        colloid_radius = self.params.colloid_radius.m_as('sim_length')

        if self.n_dims == 3:
            box_l = np.array(3 * [self.params.box_length.m_as('sim_length')])
        elif self.n_dims == 2:
            box_l = np.array(2 * [self.params.box_length.m_as('sim_length')] + [3 * colloid_radius])
        else:
            raise ValueError('we only support 2d or 3d systems')

        # system setup. Skin is a verlet list parameter that has to be set, but only affects performance
        self.system.box_l = box_l
        self.system.time_step = time_step
        self.system.cell_system.skin = 0.4
        particle_type = 0

        self.system.non_bonded_inter[particle_type, particle_type].wca.set_params(
            sigma=(2 * colloid_radius) ** (-1 / 6),
            epsilon=self.params.WCA_epsilon.m_as(
                'sim_energy'))

        # set up the particles. The handles will later be used to calculate/set forces
        rng = np.random.default_rng(self.seed)
        colloid_mass = \
            (4. / 3. * np.pi * self.params.colloid_radius ** 3 * self.params.colloid_density).m_as('sim_mass')
        colloid_rinertia = 2. / 5. * colloid_mass * colloid_radius ** 2

        for _ in range(self.params.n_colloids):
            start_pos = box_l * rng.random((3,))
            # http://mathworld.wolfram.com/SpherePointPicking.html
            theta, phi = [np.arccos(2. * rng.random() - 1), 2. * np.pi * rng.random()]
            rotation_flags = 3 * [True]
            fix_flags = 3 * [False]
            if self.n_dims == 2:
                start_pos[2] = 0
                theta = np.pi / 2.
                rotation_flags[0] = False
                rotation_flags[1] = False
                fix_flags[2] = True

            start_direction = [np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi),
                               np.cos(theta)]
            colloid = self.system.part.add(pos=start_pos,
                                           director=start_direction,
                                           mass=colloid_mass,
                                           rinertia=3 * [colloid_rinertia],
                                           rotation=rotation_flags,
                                           fix=fix_flags)
            self.colloids.append(colloid)

        self.colloid_friction_translation = 6 * np.pi * self.params.fluid_dyn_viscosity.m_as(
            'sim_dyn_viscosity') * colloid_radius
        self.colloid_friction_rotation = 8 * np.pi * self.params.fluid_dyn_viscosity.m_as(
            'sim_dyn_viscosity') * colloid_radius ** 3

        # remove overlap
        self.system.integrator.set_steepest_descent(
            f_max=0., gamma=self.colloid_friction_translation, max_displacement=0.1)
        self.system.integrator.run(1000)

        # set the brownian thermostat
        kT = (self.params.temperature * self.ureg.boltzmann_constant).m_as('sim_energy')
        self.system.thermostat.set_brownian(kT=kT,
                                            gamma=self.colloid_friction_translation,
                                            gamma_rotation=self.colloid_friction_rotation,
                                            seed=self.seed)
        self.system.integrator.set_brownian_dynamics()

        # set integrator params
        steps_per_slice = int(round(time_slice / time_step))
        self.params.steps_per_slice = steps_per_slice
        if abs(steps_per_slice - time_slice / time_step) > 1e-10:
            raise ValueError('inconsistent parameters: time_slice must be integer multiple of time_step')

    def integrate(self, n_slices, force_model: swarmrl.models.interaction_model.InteractionModel):
        for _ in range(n_slices):
            if self.system.time >= self.params.write_interval.m_as('sim_time') * self.write_idx:
                self.traj_holder['Times'].append(self.system.time)
                self.traj_holder['Unwrapped_Positions'].append(np.stack([c.pos for c in self.colloids]))
                self.traj_holder['Velocities'].append(np.stack([c.v for c in self.colloids]))
                self.traj_holder['Directors'].append(np.stack([c.director for c in self.colloids]))

                if len(self.traj_holder['Times']) >= self.write_chunk_size:
                    self._write_traj_chunk_to_file()
                    for val in self.traj_holder.values():
                        val.clear()
                    self.write_idx += 1

            self.system.integrator.run(self.params.steps_per_slice)
            for coll in self.colloids:
                coll.ext_force = force_model.calc_force(coll, [c for c in self.colloids if c is not coll])
                coll.ext_torque = force_model.calc_torque(coll, [c for c in self.colloids if c is not coll])
                new_direction = force_model.calc_new_direction(coll, [c for c in self.colloids if c is not coll])
                if new_direction is not None:
                    coll.director = new_direction

    def finalize(self):
        """
        Method to clean up after finishing the simulation (e.g. writing the last chunks of trajectory)
        """
        self._write_traj_chunk_to_file()

    def get_particle_data(self):
        return {'Unwrapped_Positions': np.stack([c.pos for c in self.colloids]),
                'Velocities': np.stack([c.v for c in self.colloids]),
                'Directors': np.stack([c.director for c in self.colloids])}

    def get_unit_system(self):
        return self.ureg
