"""
Module for the espressoMD simulations.
"""
try:
    import espressomd
    from espressomd import System, visualization
except ImportError:
    raise ImportWarning("Could not find espressomd. Features will not be available")
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
    """
    class to hold all information needed to setup and run the MD simulation.
    Provide in whichever unit you want, all quantities will be converted to simulation
    units during setup

    non-obvious attributes
    ----------------------
    time_slice:
        MD runs with internal time step of time_step. The external force/torque from
        the force_model will not be updated at every single time step, instead every
        time_slice. Therefore, time_slice must be an integer multiple of time_step.
    initiation_radius: pint length or None
        If None (default), initialize particles randomly in the whole box. If pint
        length, initialize in the center of the box withing the specified radius.
        Make sure that initiation_radius <= box_l/2.
    """

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
    initiation_radius: pint.Quantity = None


def _get_random_start_pos(
    init_radius: float, box_l: np.ndarray, rng: np.random.Generator, n_tries=100000
):
    if init_radius is None:
        return box_l * rng.random((3,))
    else:
        for _ in range(n_tries):
            start_pos = box_l * rng.random((3,))
            if np.linalg.norm(start_pos - box_l / 2.0) <= init_radius:
                return start_pos
    raise RuntimeError(f"Could not find suitable start position with {n_tries} tries")


class EspressoMD(Engine):
    """
    A class to manage the espressoMD environment.
    """

    def __init__(
        self,
        md_params,
        n_dims=3,
        seed=42,
        out_folder=".",
        loglevel=logging.DEBUG,
        write_chunk_size=100,
    ):
        """
        Constructor for the espressoMD engine.

        Parameters
        ----------
        md_params : espressomd.MDParams
                Parameter class for the espresso simulation.
        n_dims : int (default = 3)
                Number of dimensions to consider in the simulation
        seed : int
                Seed number for any generators.
        out_folder : str
                Path to an output folder to store data in. This file should have a
                reasonable amount of free space.
        loglevel : object
                Type of logging to perform.
        write_chunk_size : int
                Chunk size to use in the hdf5 writing.
        """
        self.params = md_params
        self.out_folder = out_folder
        self.seed = seed
        self.n_dims = n_dims

        self._init_unit_system()
        self._init_h5_output(write_chunk_size)
        self._init_logger(loglevel)
        self._init_calculated_quantities()

        self.system = System(box_l=3 * [1])
        self.colloids = list()

        espressomd.assert_features(["ROTATION", "EXTERNAL_FORCES"])

    def _init_unit_system(self):
        """
        Initialize the unit registry managed by pint.

        Returns
        -------
        Updates the class state.
        """
        self.ureg = self.params.ureg

        # three basis units chosen arbitrarily
        self.ureg.define("sim_length = 1e-6 meter")
        self.ureg.define("sim_time = 1 second")
        self.ureg.define(f"sim_energy = 293 kelvin * boltzmann_constant")

        # derived units
        self.ureg.define("sim_velocity = sim_length / sim_time")
        self.ureg.define("sim_mass = sim_energy / sim_velocity**2")
        self.ureg.define("sim_dyn_viscosity = sim_mass / (sim_length * sim_time)")
        self.ureg.define("sim_force = sim_mass * sim_length / sim_time**2")

    def _init_h5_output(self, write_chunk_size: int):
        """
        Initialize the hdf5 output.

        This method will create a directory for the data to be stored within. Follwing
        this, a hdf5 database is constructed for storing of the simulation data.

        Parameters
        ----------
        write_chunk_size : int
                Chunk sized used in writing.

        Returns
        -------
        Creates hdf5 database and updates class state.
        """
        self.write_chunk_size = write_chunk_size
        self.h5_filename = self.out_folder + "/trajectory.hdf5"
        os.makedirs(self.out_folder, exist_ok=True)
        self.traj_holder = {
            "Times": list(),
            "Unwrapped_Positions": list(),
            "Velocities": list(),
            "Directors": list(),
        }

        with h5py.File(self.h5_filename, "a") as h5_outfile:
            part_group = h5_outfile.require_group("colloids")
            dataset_kwargs = dict(compression="gzip")
            traj_len = write_chunk_size

            part_group.require_dataset(
                "Times",
                shape=(traj_len, 1, 1),
                maxshape=(None, 1, 1),
                dtype=float,
                **dataset_kwargs,
            )
            for name in ["Unwrapped_Positions", "Velocities", "Directors"]:
                part_group.require_dataset(
                    name,
                    shape=(traj_len, self.params.n_colloids, 3),
                    maxshape=(None, self.params.n_colloids, 3),
                    dtype=float,
                    **dataset_kwargs,
                )
        self.write_idx = 0
        self.h5_time_steps_written = 0

    def _init_logger(self, loglevel):
        """
        Initialize the system logger.

        Parameters
        ----------
        loglevel : object
                log level with which to initialize.

        Returns
        -------
        Updates logging state in the class.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        formatter = logging.Formatter(
            fmt="[%(levelname)-10s] %(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(f"{self.out_folder}/simulation_log.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(loglevel)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(loglevel)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _init_calculated_quantities(self):
        """
        Initialize computerd values to be set later.

        Returns
        -------
        Adds attributes to the class.
        """
        self.colloid_friction_translation = None
        self.colloid_friction_rotation = None

    def _write_traj_chunk_to_file(self):
        """
        Write a chunk of data to the HDF5 database

        Returns
        -------
        Adds data to the database and updates the class state.
        """
        n_new_timesteps = len(self.traj_holder["Times"])
        if n_new_timesteps == 0:
            return

        with h5py.File(self.h5_filename, "a") as h5_outfile:
            part_group = h5_outfile["colloids"]
            for key in self.traj_holder.keys():
                dataset = part_group[key]
                values = np.stack(self.traj_holder[key], axis=0)
                # save in format (time_step, n_particles, dimension)
                dataset.resize(self.h5_time_steps_written + n_new_timesteps, axis=0)
                dataset[
                    self.h5_time_steps_written : self.h5_time_steps_written
                    + n_new_timesteps,
                    ...,
                ] = values

        self.logger.debug(f"wrote {n_new_timesteps} time steps to hdf5 file")
        self.h5_time_steps_written += n_new_timesteps

    def setup_simulation(self):
        """
        Prepare the simulation box with the given parameters.

        Returns
        -------
        Update the class state.
        """
        # parameter unit conversion
        time_step = self.params.time_step.m_as("sim_time")
        # time slice: the amount of time the integrator runs before we look at the
        # configuration and change forces
        time_slice = self.params.time_slice.m_as("sim_time")
        colloid_radius = self.params.colloid_radius.m_as("sim_length")

        if self.n_dims == 3:
            box_l = np.array(3 * [self.params.box_length.m_as("sim_length")])
        elif self.n_dims == 2:
            box_l = np.array(
                2 * [self.params.box_length.m_as("sim_length")] + [3 * colloid_radius]
            )
        else:
            raise ValueError("we only support 2d or 3d systems")

        # system setup. Skin is a verlet list parameter that has to be set, but only
        # affects performance
        self.system.box_l = box_l
        self.system.time_step = time_step
        self.system.cell_system.skin = 0.4
        particle_type = 0

        init_radius = self.params.initiation_radius
        if init_radius is not None:
            init_radius = init_radius.m_as("sim_length")

        self.system.non_bonded_inter[particle_type, particle_type].wca.set_params(
            sigma=(2 * colloid_radius) * 2 ** (-1 / 6),
            epsilon=self.params.WCA_epsilon.m_as("sim_energy"),
        )

        # set up the particles. The handles will later be used to calculate/set forces
        rng = np.random.default_rng(self.seed)
        colloid_mass = (
            4.0
            / 3.0
            * np.pi
            * self.params.colloid_radius ** 3
            * self.params.colloid_density
        ).m_as("sim_mass")
        colloid_rinertia = 2.0 / 5.0 * colloid_mass * colloid_radius ** 2

        for _ in range(self.params.n_colloids):
            start_pos = _get_random_start_pos(init_radius, box_l, rng)

            if self.n_dims == 3:
                # http://mathworld.wolfram.com/SpherePointPicking.html
                theta, phi = [
                    np.arccos(2.0 * rng.random() - 1),
                    2.0 * np.pi * rng.random(),
                ]
                start_direction = [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
                colloid = self.system.part.add(
                    pos=start_pos,
                    director=start_direction,
                    mass=colloid_mass,
                    rinertia=3 * [colloid_rinertia],
                    rotation=3 * [True],
                    fix=3 * [False],
                )
            elif self.n_dims == 2:
                # initialize with body-frame = lab-frame to set correct rotation flags
                # allow all rotations to bring the particle to correct state
                colloid = self.system.part.add(
                    pos=start_pos,
                    mass=colloid_mass,
                    rinertia=3 * [colloid_rinertia],
                    fix=[False, False, True],
                    rotation=3 * [True],
                    quat=[1, 0, 0, 0],
                )

                colloid.rotate(axis=[0, 1, 0], angle=np.pi / 2.0)
                # now the particle points along the x-axis. The lab-z axis is the
                # body-frame (-x) -axis we only allow rotation around the
                # labframe-z-axis from now on
                colloid.rotation = [True, False, False]
                # now rotate in-plane to a random direction
                phi = 2 * np.pi * rng.random()
                colloid.rotate(axis=[0, 0, 1], angle=phi)
            else:
                raise ValueError("n_dims must be 2 or 3")

            self.colloids.append(colloid)

        self.colloid_friction_translation = (
            6
            * np.pi
            * self.params.fluid_dyn_viscosity.m_as("sim_dyn_viscosity")
            * colloid_radius
        )
        self.colloid_friction_rotation = (
            8
            * np.pi
            * self.params.fluid_dyn_viscosity.m_as("sim_dyn_viscosity")
            * colloid_radius ** 3
        )

        # remove overlap
        self.system.integrator.set_steepest_descent(
            f_max=0.0, gamma=self.colloid_friction_translation, max_displacement=0.1
        )
        self.system.integrator.run(1000)

        # set the brownian thermostat
        kT = (self.params.temperature * self.ureg.boltzmann_constant).m_as("sim_energy")
        self.system.thermostat.set_brownian(
            kT=kT,
            gamma=self.colloid_friction_translation,
            gamma_rotation=self.colloid_friction_rotation,
            seed=self.seed,
        )
        self.system.integrator.set_brownian_dynamics()

        # set integrator params
        steps_per_slice = int(round(time_slice / time_step))
        self.params.steps_per_slice = steps_per_slice
        if abs(steps_per_slice - time_slice / time_step) > 1e-10:
            raise ValueError(
                "inconsistent parameters: time_slice must be integer multiple of time_step"
            )

    def integrate(self, n_slices, force_model: swarmrl.InteractionModel):
        """
        Integrate the system for n_slices steps.

        Parameters
        ----------
        n_slices : int
                Number of integration steps to run.
        force_model : swarmrl.InteractionModel
                A SwarmRL interaction model to decide particle interaction rules.

        Returns
        -------
        Runs the simulation environment.
        """
        for _ in range(n_slices):
            if (
                self.system.time
                >= self.params.write_interval.m_as("sim_time") * self.write_idx
            ):
                self.traj_holder["Times"].append(
                    np.array([self.system.time])[np.newaxis, :]
                )
                self.traj_holder["Unwrapped_Positions"].append(
                    np.stack([c.pos for c in self.colloids], axis=0)
                )
                self.traj_holder["Velocities"].append(
                    np.stack([c.v for c in self.colloids], axis=0)
                )
                self.traj_holder["Directors"].append(
                    np.stack([c.director for c in self.colloids], axis=0)
                )
                self.write_idx += 1

                if len(self.traj_holder["Times"]) >= self.write_chunk_size:
                    self._write_traj_chunk_to_file()
                    for val in self.traj_holder.values():
                        val.clear()

            for coll in self.colloids:
                other_colloids = [c for c in self.colloids if c is not coll]
                # update the state of an active learner, ignored by non ML models.
                force_model.compute_state(coll, other_colloids)
                action = force_model.calc_action(coll, other_colloids)
                coll.swimming = {"f_swim": action.force}
                coll.ext_torque = action.torque
                new_direction = action.new_direction
                if new_direction is not None:
                    if self.n_dims == 3:
                        coll.director = new_direction
                    else:
                        old_direction = coll.director
                        rotation_angle = np.arccos(np.dot(new_direction, old_direction))
                        if rotation_angle > 1e-6:
                            rotation_axis = np.cross(old_direction, new_direction)
                            rotation_axis /= np.linalg.norm(rotation_axis)
                            # only values of [0,0,1], [0,0,-1] can come out here,
                            # plusminus numerical errors
                            rotation_axis = [0, 0, round(rotation_axis[2])]
                            coll.rotate(axis=rotation_axis, angle=rotation_angle)

            self.system.integrator.run(self.params.steps_per_slice)

    def finalize(self):
        """
        Method to clean up after finishing the simulation

        Method will write the last chunks of trajectory
        """
        self._write_traj_chunk_to_file()

    def get_particle_data(self):
        """
        Collect specific particle information from the colloids.

        Returns
        -------
        information : dict
                A dict of information for all of the colloids in the system including
                unwrapped positions, velocities, and the directors of the colloids.
        """
        return {
            "Unwrapped_Positions": np.stack([c.pos for c in self.colloids]),
            "Velocities": np.stack([c.v for c in self.colloids]),
            "Directors": np.stack([c.director for c in self.colloids]),
        }

    def get_unit_system(self):
        """
        Collect the pin unit registry.

        Returns
        -------
        unit_registry: object
                The class unit registry.
        """
        return self.ureg
