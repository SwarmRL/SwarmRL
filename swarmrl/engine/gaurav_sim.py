import dataclasses
import pathlib
import shelve
import typing

import h5py
import numba
import numpy as np
import pint
import scipy.integrate

from swarmrl.engine import Engine


@dataclasses.dataclass
class GauravSimParams:
    """
    time_step < snapshot_interval < time_slice, all multiples of time_step.

    capillary_force_data_path: path to the database that contains precalculated
        force and torque data. Specify path **without** the file extension.
        The shelve module will find all files needed to load the database.
    """

    ureg: pint.UnitRegistry
    box_length: float
    time_step: float
    time_slice: float
    snapshot_interval: float
    raft_radius: float
    raft_repulsion_strength: float
    dynamic_viscosity: float
    fluid_density: float
    lubrication_threshold: float
    magnetic_constant: float
    capillary_force_data_path: pathlib.Path


def setup_unit_system(ureg: pint.UnitRegistry):
    """
    In-place definition of the unit system for the simulation,
    added to the ureg object.
    """
    # basis units
    ureg.define("sim_length = 1 micrometer")
    ureg.define("sim_time = 1 second")
    ureg.define("sim_current = 1 ampere")
    ureg.define("sim_force = 1e-6 newton")

    # gaurav's original unit system isn't consistent, but this should
    # only affect the water density

    # derived units
    ureg.define("sim_mass = sim_force * sim_time**2 / sim_length")
    ureg.define("sim_velocity = sim_length / sim_time")
    ureg.define("sim_angular_velocity = 1 / sim_time")
    ureg.define("sim_dyn_viscosity = sim_mass / (sim_length * sim_time)")
    ureg.define("sim_kin_viscosity = sim_length**2 / sim_time")
    ureg.define("sim_torque = sim_length * sim_force")
    ureg.define("sim_magnetic_field = sim_mass / sim_time**2 / sim_current")
    ureg.define("sim_magnetic_permeability = sim_force / sim_current **2")
    ureg.define("sim_magnetic_moment = sim_current * sim_length**2")

    return ureg


def convert_params_to_simunits(params: GauravSimParams):
    ureg = params.ureg
    params_simunits = GauravSimParams(
        ureg=ureg,
        box_length=params.box_length.m_as("sim_length"),
        time_step=params.time_step.m_as("sim_time"),
        time_slice=params.time_slice.m_as("sim_time"),
        snapshot_interval=params.snapshot_interval.m_as("sim_time"),
        raft_radius=params.raft_radius.m_as("sim_length"),
        raft_repulsion_strength=params.raft_repulsion_strength.m_as("sim_force"),
        dynamic_viscosity=params.dynamic_viscosity.m_as("sim_dyn_viscosity"),
        fluid_density=params.fluid_density.m_as("sim_mass / sim_length**3"),
        lubrication_threshold=params.lubrication_threshold.m_as("sim_length"),
        magnetic_constant=params.magnetic_constant.m_as("sim_magnetic_permeability"),
        capillary_force_data_path=params.capillary_force_data_path,
    )
    return params_simunits


@dataclasses.dataclass
class Raft:
    pos: np.array
    alpha: float  # in radians
    magnetic_moment: float

    def get_director(self) -> np.array:
        return np.array([np.cos(self.alpha), np.sin(self.alpha)])


@dataclasses.dataclass
class GauravAction:
    amplitudes: np.array
    frequencies: np.array
    phases: np.array
    offsets: np.array


class Model:
    def calc_action(self, rafts: typing.List[Raft]) -> GauravAction:
        raise NotImplementedError


class ConstantAction(Model):
    def __init__(self, action: GauravAction) -> None:
        self.action = action

    def calc_action(self, rafts: typing.List[Raft]) -> GauravAction:
        return self.action


def calc_B_field(action: GauravAction, t: float):
    angles = 2 * np.pi * action.frequencies * t + action.phases
    return action.amplitudes * np.sin(angles) + action.offsets


@numba.jit
def calc_f_dipole(
    r_vec: np.array,
    phi_1: float,
    phi_2: float,
    m_1: float,
    m_2: float,
    mag_const: float,
):
    """
    eq. 42, "AN ANALYTIC SOLUTION FOR THE FORCE BETWEEN TWO MAGNETIC DIPOLES"
    """
    m1_vec = m_1 * np.array([np.cos(phi_1), np.sin(phi_1), 0])
    m2_vec = m_2 * np.array([np.cos(phi_2), np.sin(phi_2), 0])
    r = np.linalg.norm(r_vec)
    r_hat = np.zeros(3)
    r_hat[0] = r_vec[0] / r
    r_hat[1] = r_vec[1] / r
    f = (
        3
        * mag_const
        / (4 * np.pi * r**4)
        * (
            np.cross(np.cross(r_hat, m1_vec), m2_vec)
            + np.cross(np.cross(r_hat, m2_vec), m1_vec)
            - 2 * np.dot(m1_vec, m2_vec) * r_hat
            + 5 * r_hat * np.dot(np.cross(r_hat, m1_vec), np.cross(r_hat, m2_vec))
        )
    )
    return f[:2]


@numba.jit
def calc_lubG(x, radius):
    if x == 0:
        return 0.0212758 / radius**3
    else:
        logx = np.log(x)
        return ((0.0212758 * (-logx) + 0.181089) * (-logx) + 0.381213) / (
            radius**3 * ((-logx) * ((-logx) + 6.0425) + 6.32549)
        )


@numba.jit
def calc_lubA(x, radius):
    if x == 0:
        return 0
    else:
        return x * (-0.285524 * x + 0.095493 * x * np.log(x) + 0.106103) / radius


@numba.jit
def calc_lubB(x: float, radius: float):
    if x == 0.0:
        return 0.0212764 / radius
    else:
        logx = np.log(x)
        return ((0.0212764 * (-logx) + 0.157378) * (-logx) + 0.269886) / (
            radius * (-logx) * ((-logx) + 6.0425) + 6.32549
        )


@numba.jit
def calc_lubC(x, radius):
    return -radius * calc_lubG(
        x, radius
    )  # Gaurav: in the paper that is not true (eq 13, 14)


@numba.jit
def calc_f_mag_on(r_ij, phi_ij, prefactor):
    # eq 28
    return prefactor / r_ij**4 * (1 - 3 * np.cos(phi_ij) ** 2)


@numba.jit
def calc_f_mag_off(r_ij, phi_ij, prefactor):
    # eq 29
    return prefactor / r_ij**4 * 2 * np.cos(phi_ij) * np.sin(phi_ij)


class GauravSim(Engine):
    """
    Based on Gaurav's implementation of raft motion described in
    "Order and information in the patterns of spinning magnetic
    micro-disks at the air-water interface"
    doi:10.1126/sciadv.abk0685
    """

    def __init__(
        self,
        params: GauravSimParams,
        out_folder: typing.Union[str, pathlib.Path],
        h5_group_tag: str = "rafts",
        with_precalc_capillary: bool = True,
    ):
        setup_unit_system(params.ureg)
        self.params: GauravSimParams = convert_params_to_simunits(params)
        self.rafts: typing.List[Raft] = []

        self.out_folder: pathlib.Path = pathlib.Path(out_folder).resolve()
        self.h5_group_tag = h5_group_tag
        self.with_precalc_capillary = with_precalc_capillary

        self.integration_initialised = False
        self.slice_idx = None
        self.current_action: GauravAction = None

        if self.with_precalc_capillary:
            cap_force, cap_torque, cap_distances = self._load_cap_forces()
            self.capillary_force = cap_force
            self.capillary_torque = cap_torque
            self.max_cap_distance = cap_distances[-1]

    def add_raft(self, pos: np.array, alpha: float, magnetic_moment: float):
        self._check_already_initialised()
        self.rafts.append(
            Raft(
                pos.m_as("sim_length"),
                alpha,
                magnetic_moment.m_as("sim_magnetic_moment"),
            )
        )

    def _get_state_from_rafts(self):
        return np.array([[r.pos[0], r.pos[1], r.alpha] for r in self.rafts])

    def _update_rafts_from_state(self, state):
        for r, s in zip(self.rafts, state):
            r.pos = s[:2]
            r.alpha = s[2]

    def _check_already_initialised(self):
        if self.integration_initialised:
            raise RuntimeError(
                "You cannot change the system configuration "
                "after the first call to integrate()"
            )

    def _init_h5_output(self):
        """
        Initialize the hdf5 output.

        This method will create a directory for the data to be stored within. Follwing
        this, a hdf5 database is constructed for storing of the simulation data.
        """
        self.h5_filename = self.out_folder / "trajectory.hdf5"
        self.out_folder.mkdir(parents=True, exist_ok=True)

        n_rafts = len(self.rafts)

        self.h5_dataset_keys = ["Times", "Ids", "Alphas", "Unwrapped_Positions"]

        # create datasets with 3 dimension regardless of data dimension to make
        # data handling easier later
        with h5py.File(self.h5_filename.as_posix(), "a") as h5_outfile:
            part_group = h5_outfile.require_group(self.h5_group_tag)
            dataset_kwargs = dict(compression="gzip")

            part_group.require_dataset(
                "Times",
                shape=(0, 1, 1),
                maxshape=(None, 1, 1),
                dtype=float,
                **dataset_kwargs,
            )
            for name in ["Ids"]:
                part_group.require_dataset(
                    name,
                    shape=(0, n_rafts, 1),
                    maxshape=(None, n_rafts, 1),
                    dtype=int,
                    **dataset_kwargs,
                )
            for name in ["Alphas"]:
                part_group.require_dataset(
                    name,
                    shape=(0, n_rafts, 1),
                    maxshape=(None, n_rafts, 1),
                    dtype=float,
                    **dataset_kwargs,
                )
            for name in ["Unwrapped_Positions"]:
                part_group.require_dataset(
                    name,
                    shape=(0, n_rafts, 2),
                    maxshape=(None, n_rafts, 2),
                    dtype=float,
                    **dataset_kwargs,
                )

    def write_to_h5(self, traj_state_flat: np.array, times: np.array):
        # traj_state_flat.shape = (n_part*3, n_snapshots)
        n_new_snapshots = len(times)
        n_partcls = traj_state_flat.shape[0] // 3

        traj_state_h5format = traj_state_flat.T
        traj_state_h5format = np.reshape(
            traj_state_h5format, (n_new_snapshots, n_partcls, 3)
        )

        # add dimensions to low-dim arrays
        write_chunk = {
            "Times": times[:, None, None],
            "Ids": np.repeat(np.arange(n_partcls)[None, :], n_new_snapshots, axis=0)[
                :, :, None
            ],
            "Alphas": traj_state_h5format[:, :, 2][:, :, None],
            "Unwrapped_Positions": traj_state_h5format[:, :, :2],
        }

        with h5py.File(self.h5_filename, "a") as h5_outfile:
            part_group = h5_outfile[self.h5_group_tag]
            for key, values in write_chunk.items():
                dataset = part_group[key]
                n_snapshots_old = dataset.shape[0]
                dataset.resize(n_snapshots_old + n_new_snapshots, axis=0)
                dataset[
                    n_snapshots_old : n_snapshots_old + n_new_snapshots,
                    ...,
                ] = values

    def _load_cap_forces(self):
        with shelve.open(self.params.capillary_force_data_path.as_posix()) as tempShelf:
            capillaryEEDistances = tempShelf["eeDistanceCombined"]  # unit: m
            capillaryForcesDistancesAsRowsLoaded = tempShelf[
                "forceCombinedDistancesAsRowsAll360"
            ]  # unit: N
            capillaryTorquesDistancesAsRowsLoaded = tempShelf[
                "torqueCombinedDistancesAsRowsAll360"
            ]  # unit: N.m

        # convert units
        capillaryEEDistances = self.params.ureg.Quantity(
            capillaryEEDistances, "meter"
        ).m_as("sim_length")
        capillaryForcesDistancesAsRowsLoaded = self.params.ureg.Quantity(
            capillaryForcesDistancesAsRowsLoaded, "newton"
        ).m_as("sim_force")
        capillaryTorquesDistancesAsRowsLoaded = self.params.ureg.Quantity(
            capillaryTorquesDistancesAsRowsLoaded, "newton * meter"
        ).m_as("sim_torque")

        # further data treatment on capillary force profile
        # insert the force and torque at eeDistance = 1um
        # as the value for eedistance = 0um.
        capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
        capillaryForcesDistancesAsRows = np.concatenate(
            (
                capillaryForcesDistancesAsRowsLoaded[:1, :],
                capillaryForcesDistancesAsRowsLoaded,
            ),
            axis=0,
        )
        capillaryTorquesDistancesAsRows = np.concatenate(
            (
                capillaryTorquesDistancesAsRowsLoaded[:1, :],
                capillaryTorquesDistancesAsRowsLoaded,
            ),
            axis=0,
        )

        # add angle=360, the same as angle = 0
        capillaryForcesDistancesAsRows = np.concatenate(
            (
                capillaryForcesDistancesAsRows,
                capillaryForcesDistancesAsRows[:, 0].reshape(1001, 1),
            ),
            axis=1,
        )
        capillaryTorquesDistancesAsRows = np.concatenate(
            (
                capillaryTorquesDistancesAsRows,
                capillaryTorquesDistancesAsRows[:, 0].reshape(1001, 1),
            ),
            axis=1,
        )

        # correct for the negative sign of the torque
        capillaryTorquesDistancesAsRows = -capillaryTorquesDistancesAsRows

        # some extra treatment for the force matrix
        # note the sharp transition at the peak-peak position (45 deg):
        # only 1 deg difference,
        nearEdgeSmoothingThres = (
            1  # unit: micron; if 1, then it is equivalent to no smoothing.
        )
        for distanceToEdge in np.arange(nearEdgeSmoothingThres):
            capillaryForcesDistancesAsRows[distanceToEdge, :] = (
                capillaryForcesDistancesAsRows[nearEdgeSmoothingThres, :]
            )
            capillaryTorquesDistancesAsRows[distanceToEdge, :] = (
                capillaryTorquesDistancesAsRows[nearEdgeSmoothingThres, :]
            )

        # select a cut-off distance below which all the attractive force
        # (negative-valued) becomes zero,due to raft wall-wall repulsion
        capAttractionZeroCutoff = 0
        mask = np.concatenate(
            (
                capillaryForcesDistancesAsRows[:capAttractionZeroCutoff, :] < 0,
                np.zeros(
                    (
                        capillaryForcesDistancesAsRows.shape[0]
                        - capAttractionZeroCutoff,
                        capillaryForcesDistancesAsRows.shape[1],
                    ),
                    dtype=int,
                ),
            ),
            axis=0,
        )
        capillaryForcesDistancesAsRows[mask.nonzero()] = 0

        # realign the first peak-peak direction with an
        # angle = capillaryPeakOffset from the x-axis.
        capillaryPeakOffset = 0
        capillaryForcesDistancesAsRows = np.roll(
            capillaryForcesDistancesAsRows, capillaryPeakOffset, axis=1
        )  # 45 is due to original data
        capillaryTorquesDistancesAsRows = np.roll(
            capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis=1
        )

        return (
            capillaryForcesDistancesAsRows,
            capillaryTorquesDistancesAsRows,
            capillaryEEDistances,
        )

    def _calc_rot_mobility(self, r_ij):
        edge_edge_ij = r_ij - 2 * self.params.raft_radius
        if edge_edge_ij >= self.params.lubrication_threshold:
            return 1 / (
                8 * np.pi * self.params.dynamic_viscosity * self.params.raft_radius**3
            )
        else:
            x = max([0, edge_edge_ij / self.params.raft_radius])
            return calc_lubG(x, self.params.raft_radius) / self.params.dynamic_viscosity

    def _calc_trans_mobility(self, r_ij, lub_option):
        edge_edge_ij = r_ij - 2 * self.params.raft_radius
        if edge_edge_ij > self.params.lubrication_threshold or lub_option == "none":
            return 1 / (
                6 * np.pi * self.params.dynamic_viscosity * self.params.raft_radius
            )
        else:
            x = max([0, edge_edge_ij / self.params.raft_radius])
            if lub_option == "A":
                lub = calc_lubA(x, self.params.raft_radius)
            elif lub_option == "B":
                lub = calc_lubB(x, self.params.raft_radius)
            elif lub_option == "C":
                lub = calc_lubC(x, self.params.raft_radius)
            else:
                raise ValueError(f"Unknown lub option {lub_option}")
            return lub / self.params.dynamic_viscosity

    def _get_omega(self, state, b_field):
        b_field_angle = np.arctan2(b_field[1], b_field[0])
        b_field_norm = np.linalg.norm(b_field)
        mag_moments = np.array([r.magnetic_moment for r in self.rafts])
        b_field_torque = (
            mag_moments * b_field_norm * np.sin(b_field_angle - state[:, 2])
        )

        n_rafts = len(state)
        omegas = np.zeros(n_rafts)
        mag_moments = [r.magnetic_moment for r in self.rafts]
        for i in range(n_rafts):
            r_ijs = state[:, :2] - state[i, :2][None, :]
            r_ij_norms = np.linalg.norm(r_ijs, axis=1)
            r_ij_norms[i] = np.inf
            edge_edge_dists = r_ij_norms - 2 * self.params.raft_radius
            for j in range(i + 1, n_rafts):
                r_ij_vec = r_ijs[j, :]
                r_ij_norm = r_ij_norms[j]
                r_ij_angle = np.arctan2(r_ij_vec[1], r_ij_vec[0])
                phi_i = state[i, 2] - r_ij_angle
                phi_j = state[j, 2] - r_ij_angle
                # eq. 31
                torque_dipole_dipole = (
                    self.params.magnetic_constant
                    * mag_moments[i]
                    * mag_moments[j]
                    / (4 * np.pi * r_ij_norm**3)
                    * (3 * np.cos(phi_j) * np.sin(phi_i) + np.sin(phi_i - phi_j))
                )
                rot_mob = self._calc_rot_mobility(r_ij_norm)
                omegas[i] += rot_mob * torque_dipole_dipole
                omegas[j] += -rot_mob * torque_dipole_dipole
                phi_ij = phi_i
                if (
                    self.with_precalc_capillary
                    and edge_edge_dists[j] < self.max_cap_distance
                ):
                    angle_idx = int(360 / np.pi * phi_ij + 0.5) % 360
                    dist_idx = max(0, int(edge_edge_dists[j] + 0.5))
                    torque_capill = self.capillary_torque[dist_idx, angle_idx]
                else:
                    torque_capill = 0.0
                omegas[i] += rot_mob * torque_capill
                omegas[j] += -rot_mob * torque_capill
            omegas[i] += (
                self._calc_rot_mobility(np.min(edge_edge_dists)) * b_field_torque[i]
            )

        return omegas

    def _get_vel(self, state, omegas, b_field):
        n_rafts = len(state)
        vels = np.zeros((n_rafts, 2))
        mag_moments = [r.magnetic_moment for r in self.rafts]

        # single particle forces (wall repulsion)
        for i in range(n_rafts):
            # distance to bottom left corner = the coordinates
            dist_bot_left = state[i, :2]
            # distance to top right corner
            dist_top_right = self.params.box_length - dist_bot_left
            prefactor = (
                self._calc_trans_mobility(
                    0,
                    "none",
                )
                * self.params.fluid_density
                * omegas[i] ** 2
                * self.params.raft_radius**7
            )
            vels[i, :] += prefactor * (1 / dist_bot_left**3 - 1 / dist_top_right**3)

        # pair interactions
        for i in range(n_rafts):
            r_ijs = state[:, :2] - state[i, :2][None, :]
            r_ij_norms = np.linalg.norm(r_ijs, axis=1)
            r_ij_norms[i] = np.inf
            edge_edge_dists = r_ij_norms - 2 * self.params.raft_radius
            for j in range(i + 1, n_rafts):
                r_ij_vec = r_ijs[j, :]
                r_ij_norm = r_ij_norms[j]
                r_ij_angle = np.arctan2(r_ij_vec[1], r_ij_vec[0])
                r_ij_unit_vec = r_ij_vec / r_ij_norm
                r_ij_cross_z = np.array([r_ij_unit_vec[1], -r_ij_unit_vec[0]])
                edge_edge_dist = edge_edge_dists[j]
                phi_i = state[i, 2] - r_ij_angle
                phi_ij = phi_i

                # f_mag_on_prefactor = (
                #     3
                #     * self.params.magnetic_constant
                #     * mag_moments[i]
                #     * mag_moments[j]
                #     / (4 * np.pi)
                # )
                # f_mag_off_prefactor = f_mag_on_prefactor
                # f_mag_on = self.calc_f_mag_on(r_ij_norm, phi_ij, f_mag_on_prefactor)
                # f_mag_off =
                # self.calc_f_mag_off(r_ij_norm, phi_ij, f_mag_off_prefactor)

                f_dip_dip = calc_f_dipole(
                    max(r_ij_norm, 2 * self.params.raft_radius) * r_ij_unit_vec,
                    state[i, 2],
                    state[j, 2],
                    mag_moments[i],
                    mag_moments[j],
                    self.params.magnetic_constant,
                )
                f_mag_on = np.dot(f_dip_dip, r_ij_unit_vec)
                f_mag_off = np.dot(f_dip_dip, r_ij_cross_z)

                if (
                    self.with_precalc_capillary
                    and edge_edge_dist < self.max_cap_distance
                ):
                    angle_idx = int(360 / np.pi * phi_ij + 0.5) % 360
                    dist_idx = max(0, int(edge_edge_dist + 0.5))
                    f_capillary = self.capillary_force[dist_idx, angle_idx]
                else:
                    f_capillary = 0.0

                r_7_force = (
                    self.params.fluid_density
                    * omegas[j] ** 2
                    * self.params.raft_radius**7
                    / r_ij_norm**3
                )  # Gaurav: why this omega and not the other?
                mob_A = self._calc_trans_mobility(r_ij_norm, "A")
                mob_B = self._calc_trans_mobility(r_ij_norm, "B")

                vel_ij_along_r_ij = (
                    mob_A * (f_mag_on + f_capillary + r_7_force) * r_ij_unit_vec
                )
                if edge_edge_dist < 0:
                    vel_ij_along_r_ij += (
                        self._calc_trans_mobility(0, "none")
                        * self.params.raft_repulsion_strength
                        * edge_edge_dist  # pretty sure this has to be positive
                        / self.params.raft_radius
                    )

                vel_ij_along_r_ij_cross_z = mob_B * f_mag_off
                if edge_edge_dist >= self.params.lubrication_threshold:
                    r_3_vel = (
                        -self.params.raft_radius**3 * omegas[i] / r_ij_norm**2
                    )  # Gaurav: also here which omega
                    vel_ij_along_r_ij_cross_z += r_3_vel
                elif self.params.lubrication_threshold > edge_edge_dist > 0:
                    mob_C = self._calc_trans_mobility(r_ij_norm, "C")
                    b_field_angle = np.arctan2(b_field[1], b_field[0])
                    vel_ij_along_r_ij_cross_z += (
                        mob_C
                        * mag_moments[i]
                        * np.linalg.norm(b_field)
                        * np.sin(b_field_angle - state[i, 2])
                    )
                else:
                    pass

                vels[i, :] += vel_ij_along_r_ij * r_ij_unit_vec
                vels[i, :] += vel_ij_along_r_ij_cross_z * r_ij_cross_z
                vels[j, :] += -vel_ij_along_r_ij * r_ij_unit_vec
                vels[j, :] += -vel_ij_along_r_ij_cross_z * r_ij_cross_z

        return vels

    def get_rhs(self, t, state_flat: np.array):
        state = state_flat.reshape((-1, 3))
        b_field = calc_B_field(self.current_action, t)
        omega = self._get_omega(state, b_field)
        vel = self._get_vel(state, omega, b_field)
        state_derivative = np.concatenate([vel, omega[:, None]], axis=1)
        return state_derivative.reshape(-1)

    def integrate(self, n_slices: int, model: Model):

        if not self.integration_initialised:
            self.time = 0.0
            self.slice_idx = 0
            self._init_h5_output()
            self.integration_initialised = True

        def rhs(t, state):
            # get_rhs has self as the first argument so we write this
            # small wrapper to remove it from the call signature
            return self.get_rhs(t, state)

        state_flat = self._get_state_from_rafts().flatten()

        n_snapshots_per_slice = int(
            round(self.params.time_slice / self.params.snapshot_interval)
        )
        for _ in range(n_slices):
            # Check for a model termination condition.
            if model.kill_switch:
                break
            self.current_action = model.calc_action(self.rafts)
            sol = scipy.integrate.solve_ivp(
                rhs,
                (self.time, self.time + self.params.time_slice),
                state_flat,
                method="RK23",
                first_step=self.params.time_step,
                max_step=self.params.time_step,
                min_step=self.params.time_step,
                t_eval=np.linspace(
                    self.time + self.params.snapshot_interval,
                    self.time + self.params.time_slice,
                    num=n_snapshots_per_slice,
                ),
            )
            if not sol.success:
                raise RuntimeError(
                    f"Integration crashed at time {self.time}. Reason: {sol.message}"
                )
            self.write_to_h5(sol.y, sol.t)

            self.time = sol.t[-1]
            state_flat = sol.y[:, -1]
            self._update_rafts_from_state(state_flat.reshape((-1, 3)))
