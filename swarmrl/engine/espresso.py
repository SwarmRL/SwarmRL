"""
Module for the espressoMD simulations.
"""
import dataclasses
import logging
import os
import typing

import h5py
import numpy as np
import pint

import swarmrl.models.interaction_model

from .engine import Engine

logger = logging.getLogger(__name__)
try:
    import espressomd
    import espressomd.constraints
    import espressomd.shapes
except ModuleNotFoundError:
    logger.warning("Could not find espressomd. Features will not be available")


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
    """

    ureg: pint.UnitRegistry
    box_length: pint.Quantity
    fluid_dyn_viscosity: pint.Quantity
    WCA_epsilon: pint.Quantity
    temperature: pint.Quantity
    time_step: pint.Quantity
    time_slice: pint.Quantity
    write_interval: pint.Quantity


def _get_random_angles(rng: np.random.Generator):
    # https://mathworld.wolfram.com/SpherePointPicking.html
    return np.arccos(2.0 * rng.random() - 1), 2.0 * np.pi * rng.random()


def _vector_from_angles(theta, phi):
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def _get_random_start_pos(
    init_radius: float, init_center: np.array, dim: int, rng: np.random.Generator
):
    if dim == 2:
        r = init_radius * np.sqrt(rng.random())
        theta = 2 * np.pi * rng.random()
        pos = r * np.array([np.cos(theta), np.sin(theta), 0])
        assert init_center[2] == 0.0
    elif dim == 3:
        r = init_radius * np.cbrt(rng.random())
        pos = r * _vector_from_angles(*_get_random_angles(rng))
    else:
        raise ValueError("Random position finder only implemented for 2d and 3d")

    return pos + init_center


def _calc_friction_coefficients(
    dyn_visc: float, radius: float
) -> typing.Tuple[float, float]:
    particle_gamma_translation = 6 * np.pi * dyn_visc * radius
    particle_gamma_rotation = 8 * np.pi * dyn_visc * radius**3
    return particle_gamma_translation, particle_gamma_rotation


class EspressoMD(Engine):
    """
    A class to manage the espressoMD environment.

    Methods are allowed to add particles until the first call to integrate().

    Dev Note: These methods need to register the new particles
    in self.colloid_radius_register
    The first call to integrate() will then setup the interactions and database output.
    """

    def __init__(
        self,
        md_params,
        n_dims=3,
        seed=42,
        out_folder=".",
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
        write_chunk_size : int
                Chunk size to use in the hdf5 writing.
        """
        self.params = md_params
        self.out_folder = out_folder
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        if n_dims not in [2, 3]:
            raise ValueError("Only 2d and 3d are allowed")
        self.n_dims = n_dims

        self._init_unit_system()
        self.write_chunk_size = write_chunk_size

        self.system = espressomd.System(box_l=3 * [1.0])
        self._init_system()
        self.colloids = list()

        # register to lookup which type has which radius
        self.colloid_radius_register = {}

        # after the first call to integrate, no more changes to the engine are allowed
        self.integration_initialised = False

        espressomd.assert_features(
            ["ROTATION", "EXTERNAL_FORCES", "THERMOSTAT_PER_PARTICLE"]
        )

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
        self.ureg.define("sim_energy = 293 kelvin * boltzmann_constant")

        # derived units
        self.ureg.define("sim_velocity = sim_length / sim_time")
        self.ureg.define("sim_mass = sim_energy / sim_velocity**2")
        self.ureg.define("sim_dyn_viscosity = sim_mass / (sim_length * sim_time)")
        self.ureg.define("sim_force = sim_mass * sim_length / sim_time**2")

    def _init_system(self):
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

        box_l = np.array(3 * [self.params.box_length.m_as("sim_length")])

        # system setup. Skin is a verlet list parameter that has to be set, but only
        # affects performance
        self.system.box_l = box_l
        self.system.time_step = time_step
        self.system.cell_system.skin = 0.4

        # set integrator params
        steps_per_slice = int(round(time_slice / time_step))
        self.params.steps_per_slice = steps_per_slice
        if abs(steps_per_slice - time_slice / time_step) > 1e-10:
            raise ValueError(
                "inconsistent parameters: time_slice must be integer multiple of"
                " time_step"
            )

    def _rotate_colloid_to_2d(self, colloid, theta):
        # rotate around the y-axis by 90 degrees
        colloid.rotate(axis=[0, 1, 0], angle=np.pi / 2.0)
        # now the particle points along the x-axis. The lab-z axis is the
        # body-frame (-x) -axis. We only allow rotation around the
        # labframe-z-axis from now on
        colloid.rotation = [True, False, False]
        # now rotate in-plane
        colloid.rotate(axis=[0, 0, 1], angle=theta)

    def _check_already_initialised(self):
        if self.integration_initialised:
            raise RuntimeError(
                "You cannot change the system configuration "
                "after the first call to integrate()"
            )

    def add_colloids(
        self,
        n_colloids: int,
        radius_colloid: pint.Quantity,
        random_placement_center: pint.Quantity,
        random_placement_radius: pint.Quantity,
        type_colloid=0,
    ):
        """
        Parameters
        ----------
        n_colloids
        radius_colloid
        random_placement_center
        random_placement_radius
        type_colloid
            The colloids created from this method call will have this type.
            Multiple calls can be made with the same type_colloid.
            Interaction models need to be made aware if there are different types
            of colloids in the system if specific behaviour is desired.

        Returns
        -------

        """

        self._check_already_initialised()

        radius_simunits = radius_colloid.m_as("sim_length")
        init_center = random_placement_center.m_as("sim_length")
        init_rad = random_placement_radius.m_as("sim_length")

        (
            particle_gamma_translation,
            particle_gamma_rotation,
        ) = _calc_friction_coefficients(
            self.params.fluid_dyn_viscosity.m_as("sim_dyn_viscosity"), radius_simunits
        )

        for i in range(n_colloids):
            start_pos = _get_random_start_pos(
                init_rad, init_center, self.n_dims, self.rng
            )

            if self.n_dims == 3:
                colloid = self.system.part.add(
                    pos=start_pos,
                    director=_vector_from_angles(*_get_random_angles(self.rng)),
                    rotation=3 * [True],
                    gamma=particle_gamma_translation,
                    gamma_rot=particle_gamma_rotation,
                    fix=3 * [False],
                    type=type_colloid,
                )
            else:
                # initialize with body-frame = lab-frame to set correct rotation flags
                # allow all rotations to bring the particle to correct state
                start_angle = 2 * np.pi * self.rng.random()
                colloid = self.system.part.add(
                    pos=start_pos,
                    fix=[False, False, True],
                    rotation=3 * [True],
                    gamma=particle_gamma_translation,
                    gamma_rot=particle_gamma_rotation,
                    quat=[1, 0, 0, 0],
                    type=type_colloid,
                )
                self._rotate_colloid_to_2d(colloid, start_angle)

            self.colloids.append(colloid)

        self.colloid_radius_register.update({type_colloid: radius_simunits})

    def add_colloid_precisely(
        self,
        radius_colloid: pint.Quantity,
        init_position: pint.Quantity,
        init_3D_direction: np.array = [1, 0, 0],
        init_2D_angle: float = 0,
        type_colloid=0,
    ):
        """
        Parameters
        ----------
        radius_colloid
        init_position
        init_3D_direction
        init_2D_angle
        type_colloid
            The colloids created from this method call will have this type.
            Multiple calls can be made with the same type_colloid.
            Interaction models need to be made aware if there are different types
            of colloids in the system if specific behaviour is desired.

        Returns
        -------

        """

        self._check_already_initialised()

        radius_simunits = radius_colloid.m_as("sim_length")
        init_center = init_position.m_as("sim_length")
        init_3D_direction = init_3D_direction / np.linalg.norm(init_3D_direction)

        (
            particle_gamma_translation,
            particle_gamma_rotation,
        ) = _calc_friction_coefficients(
            self.params.fluid_dyn_viscosity.m_as("sim_dyn_viscosity"), radius_simunits
        )

        if self.n_dims == 3:
            colloid = self.system.part.add(
                pos=init_center,
                director=init_3D_direction,
                rotation=3 * [True],
                gamma=particle_gamma_translation,
                gamma_rot=particle_gamma_rotation,
                fix=3 * [False],
                type=type_colloid,
            )
        else:
            # initialize with body-frame = lab-frame to set correct rotation flags
            # allow all rotations to bring the particle to correct state
            start_angle = np.fmod(init_2D_angle, 2 * np.pi)
            init_center[2] = 0  # get rid of z-coordinate in 2D coordinates
            start_pos = init_center
            colloid = self.system.part.add(
                pos=start_pos,
                fix=[False, False, True],
                rotation=3 * [True],
                gamma=particle_gamma_translation,
                gamma_rot=particle_gamma_rotation,
                quat=[1, 0, 0, 0],
                type=type_colloid,
            )
            self._rotate_colloid_to_2d(colloid, start_angle)

        self.colloids.append(colloid)

        self.colloid_radius_register.update({type_colloid: radius_simunits})

    def add_rod(
        self,
        rod_center: pint.Quantity,
        rod_length: pint.Quantity,
        rod_thickness: pint.Quantity,
        rod_start_angle: float,
        n_particles: int,
        friction_trans: pint.Quantity,
        friction_rot: pint.Quantity,
        rod_particle_type: int,
        fixed: bool = True,
    ):
        """
        Add a rod to the system.
        A rod consists of n_particles point particles that are rigidly connected
        and rotate/move as a whole
        Parameters
        ----------
        rod_center
        rod_length
        rod_thickness
            Make sure there are enough particles.
            If the thickness is too thin, the rod might get holes
        rod_start_angle
        n_particles
            Must be uneven number such that there always is a central particle
        friction_trans
            Irrelevant if fixed==True
        friction_rot
        rod_particle_type
            The rod is made out of points so they get their own type.
        fixed
            Fixes the central particle of the rod.

        Returns
        -------
        The espresso handle to the central particle. For debugging purposes only
        """
        self._check_already_initialised()
        if self.n_dims != 2:
            raise ValueError("Rod can only be added in 2d")
        if rod_center[2].magnitude != 0:
            raise ValueError(f"Rod center z-component must be 0. You gave {rod_center}")
        if n_particles % 2 != 1:
            raise ValueError(f"n_particles must be uneven. You gave {n_particles}")

        espressomd.assert_features(["VIRTUAL_SITES_RELATIVE"])
        import espressomd.virtual_sites as evs

        self.system.virtual_sites = evs.VirtualSitesRelative(have_quaternion=True)

        center_pos = rod_center.m_as("sim_length")
        fric_trans = friction_trans.m_as("sim_force/sim_velocity")  # [F / v]
        fric_rot = friction_rot.m_as(
            "sim_force * sim_length *  sim_time"
        )  # [M / omega]
        partcl_radius = rod_thickness.m_as("sim_length") / 2

        # place the real particle
        center_part = self.system.part.add(
            pos=center_pos,
            quat=[1, 0, 0, 0],
            rotation=3 * [True],
            fix=[fixed, fixed, True],
            gamma=fric_trans,
            gamma_rot=fric_rot,
            type=rod_particle_type,
        )
        self._rotate_colloid_to_2d(center_part, rod_start_angle)
        self.colloids.append(center_part)

        # place virtual
        point_span = rod_length.m_as("sim_length") - 2 * partcl_radius
        point_dist = point_span / (n_particles - 1)
        if point_dist > 2 * partcl_radius:
            logger.warning(
                "your rod has holes. "
                f"Particle radius {partcl_radius} "
                f"particle_distance {point_dist} "
                "(both in simulation units)"
            )

        director = _vector_from_angles(np.pi / 2, rod_start_angle)

        for k in range(n_particles - 1):
            dist_to_center = (-1) ** k * (k // 2 + 1) * point_dist
            pos_virt = center_pos + dist_to_center * director
            virtual_partcl = self.system.part.add(
                pos=pos_virt, director=director, virtual=True, type=rod_particle_type
            )
            virtual_partcl.vs_auto_relate_to(center_part)
            self.colloids.append(virtual_partcl)

        self.colloid_radius_register.update({rod_particle_type: partcl_radius})

        return center_part

    def add_confining_walls(self, wall_type: int):
        """
        Walls on the edges of the box, will interact with particles through WCA.
        Is NOT communicated to the interaction models, though.

        Parameters
        ----------
        wall_type
            Wall interacts with particles, so it needs its own type.

        Returns
        -------
        """
        self._check_already_initialised()
        if wall_type in self.colloid_radius_register.keys():
            raise ValueError(
                f"wall type {wall_type} is already taken "
                "by other system component. Choose a new one"
            )

        wall_shapes = []
        wall_shapes.append(espressomd.shapes.Wall(dist=0, normal=[1, 0, 0]))
        wall_shapes.append(
            espressomd.shapes.Wall(dist=-self.system.box_l[0], normal=[-1, 0, 0])
        )
        wall_shapes.append(espressomd.shapes.Wall(dist=0, normal=[0, 1, 0]))
        wall_shapes.append(
            espressomd.shapes.Wall(dist=-self.system.box_l[1], normal=[0, -1, 0])
        )
        if self.n_dims == 3:
            wall_shapes.append(espressomd.shapes.Wall(dist=0, normal=[0, 0, 1]))
            wall_shapes.append(
                espressomd.shapes.Wall(dist=-self.system.box_l[2], normal=[0, 0, -1])
            )

        for wall_shape in wall_shapes:
            constr = espressomd.constraints.ShapeBasedConstraint(
                shape=wall_shape, particle_type=wall_type, penetrable=False
            )
            self.system.constraints.add(constr)

        # the wall itself has no radius, only the particle radius counts
        self.colloid_radius_register.update({wall_type: 0.0})

    def _setup_interactions(self):
        for type_0, rad_0 in self.colloid_radius_register.items():
            for type_1, rad_1 in self.colloid_radius_register.items():
                if type_0 > type_1:
                    continue
                self.system.non_bonded_inter[type_0, type_1].wca.set_params(
                    sigma=(rad_0 + rad_1) * 2 ** (-1 / 6),
                    epsilon=self.params.WCA_epsilon.m_as("sim_energy"),
                )

    def add_const_force_to_colloids(self, force: pint.Quantity, type: int):
        """
        Parameters
        ----------
        force: pint.Quantity
            A Quantity of numpy array, e.g. f = Quantity(np.array([1,2,3]), "newton")
        type: int
            The type of colloid that gets the force.
            Needs to be already added to the engine
        """
        force_simunits = force.m_as("sim_force")
        parts = self.system.part.select(type=type)
        if len(parts) == 0:
            raise ValueError(
                f"Particles of type {type} not added to engine. "
                f"You currently have {self.colloid_radius_register.keys()}"
            )
        parts.ext_force = force_simunits

    def get_friction_coefficients(self, type: int):
        """
        Returns both the translational and the rotational friction coefficient
        of the desired in simulation units
        """
        radius = self.colloid_radius_register.get(type, None)
        if radius is None:
            raise ValueError(
                f"cannot get friction coefficient for type {type}. Did you actually add"
                " that particle type?"
            )
        return _calc_friction_coefficients(
            self.params.fluid_dyn_viscosity.m_as("sim_dyn_viscosity"), radius
        )

    def _init_h5_output(self):
        """
        Initialize the hdf5 output.

        This method will create a directory for the data to be stored within. Follwing
        this, a hdf5 database is constructed for storing of the simulation data.


        Returns
        -------
        Creates hdf5 database and updates class state.
        """
        self.h5_filename = self.out_folder + "/trajectory.hdf5"
        os.makedirs(self.out_folder, exist_ok=True)
        self.traj_holder = {
            "Times": list(),
            "Ids": list(),
            "Types": list(),
            "Unwrapped_Positions": list(),
            "Velocities": list(),
            "Directors": list(),
        }

        n_colloids = len(self.colloids)

        with h5py.File(self.h5_filename, "a") as h5_outfile:
            part_group = h5_outfile.require_group("colloids")
            dataset_kwargs = dict(compression="gzip")
            traj_len = self.write_chunk_size

            part_group.require_dataset(
                "Times",
                shape=(traj_len, 1, 1),
                maxshape=(None, 1, 1),
                dtype=float,
                **dataset_kwargs,
            )
            for name in ["Ids", "Types"]:
                part_group.require_dataset(
                    name,
                    shape=(traj_len, n_colloids, 1),
                    maxshape=(None, n_colloids, 1),
                    dtype=int,
                    **dataset_kwargs,
                )
            for name in ["Unwrapped_Positions", "Velocities", "Directors"]:
                part_group.require_dataset(
                    name,
                    shape=(traj_len, n_colloids, 3),
                    maxshape=(None, n_colloids, 3),
                    dtype=float,
                    **dataset_kwargs,
                )
        self.write_idx = 0
        self.h5_time_steps_written = 0

    def _update_traj_holder(self):
        # need to add axes on the non-vectorial quantities
        self.traj_holder["Times"].append(np.array([self.system.time])[:, np.newaxis])
        self.traj_holder["Ids"].append(
            np.array([c.id for c in self.colloids])[:, np.newaxis]
        )
        self.traj_holder["Types"].append(
            np.array([c.type for c in self.colloids])[:, np.newaxis]
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

        logger.debug(f"wrote {n_new_timesteps} time steps to hdf5 file")
        self.h5_time_steps_written += n_new_timesteps

    def _remove_overlap(self):
        # remove overlap
        self.system.integrator.set_steepest_descent(
            f_max=0.0, gamma=0.1, max_displacement=0.1
        )
        self.system.integrator.run(1000)

        # set the brownian thermostat
        kT = (self.params.temperature * self.ureg.boltzmann_constant).m_as("sim_energy")
        # Dummy gamma values, we set them for each particle separately.
        # If we forget to do so, the simulation will explode as a gentle reminder
        self.system.thermostat.set_brownian(
            kT=kT,
            gamma=1e-20,
            gamma_rotation=1e-20,
            seed=self.seed,
            act_on_virtual=False,
        )
        self.system.integrator.set_brownian_dynamics()

    def integrate(self, n_slices, force_model: swarmrl.models.InteractionModel = None):
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

        if not self.integration_initialised:
            self._setup_interactions()
            self._remove_overlap()
            self._init_h5_output()
            self.integration_initialised = True

        for _ in range(n_slices):
            if (
                self.system.time
                >= self.params.write_interval.m_as("sim_time") * self.write_idx
            ):
                self._update_traj_holder()
                self.write_idx += 1

                if len(self.traj_holder["Times"]) >= self.write_chunk_size:
                    self._write_traj_chunk_to_file()
                    for val in self.traj_holder.values():
                        val.clear()

            swarmrl_colloids = []
            if force_model is not None:
                for col in self.colloids:
                    swarmrl_colloids.append(
                        swarmrl.models.interaction_model.Colloid(
                            pos=col.pos, director=col.director, id=col.id, type=col.type
                        )
                    )
                actions = force_model.calc_action(swarmrl_colloids)
                for action, coll in zip(actions, self.colloids):
                    coll.swimming = {"f_swim": action.force}
                    coll.ext_torque = action.torque
                    new_direction = action.new_direction
                    if new_direction is not None:
                        if self.n_dims == 3:
                            coll.director = new_direction
                        else:
                            old_direction = coll.director
                            rotation_angle = np.arccos(
                                np.dot(new_direction, old_direction)
                            )
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
            "Id": np.array([c.id for c in self.colloids]),
            "Type": np.array([c.type for c in self.colloids]),
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
