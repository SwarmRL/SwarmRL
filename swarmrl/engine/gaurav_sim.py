import dataclasses
import pathlib
import shelve
import typing

import numpy as np
import scipy

from .engine import Engine


@dataclasses.dataclass
class GauravSimParams:
    box_length: float
    n_groves: int = 6
    time_step: float
    raft_radius: float
    raft_n_groves: int
    raft_repulsion_strength: float
    dynamic_viscosity: float
    fluid_density: float
    lubrication_threshold: float
    magnetic_constant: float
    capillary_force_data_path: pathlib.Path


@dataclasses.dataclass
class Raft:
    x: float
    y: float
    pos: np.array
    alpha: float  # in radians
    magnetic_moment: float
    radius: float

    def get_director(self) -> np.array:
        return np.array([np.cos(self.alpha), np.sin(self.alpha)])


@dataclasses.dataclass
class BFieldCalculator:
    amplitudes: np.array
    frequencies: np.array
    phases: np.array
    offsets: np.array

    def calc_B_field(self, t):
        return (
            self.amplitudes * np.sin(2 * np.pi * self.frequencies * t + self.phases)
            + self.offsets
        )


class GauravSim(Engine):
    def __init__(
        self,
        params: GauravSimParams,
    ):
        self.params: GauravSimParams = params
        self.rafts: typing.List[Raft] = []
        self.bfield_calculator: BFieldCalculator = None

        cap_force, cap_torque = self._load_cap_forces()
        self.capillary_force = cap_force
        self.capillary_torque = cap_torque

    def add_raft(self, x, y, alpha):
        self.rafts.append(Raft(x, y, alpha))

    def add_bfield(self, bfield_calculator: BFieldCalculator):
        self.bfield_calculator = bfield_calculator

    def _get_state_from_rafts(self):
        return np.array([r.x, r.y, r.alpha] for r in self.rafts)

    def _load_cap_forces(self):
        if not self.params.capillary_force_data_path.exists():
            raise FileNotFoundError(
                "You must provide a path to a file that contains the precomputed"
                " capillary interaction"
            )

        with shelve.open(self.params.capillary_force_data_path) as tempShelf:
            capillaryEEDistances = tempShelf["eeDistanceCombined"]  # unit: m
            capillaryForcesDistancesAsRowsLoaded = tempShelf[
                "forceCombinedDistancesAsRowsAll360"
            ]  # unit: N
            capillaryTorquesDistancesAsRowsLoaded = tempShelf[
                "torqueCombinedDistancesAsRowsAll360"
            ]  # unit: N.m

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

        return capillaryForcesDistancesAsRows, capillaryTorquesDistancesAsRows

    def _calc_rot_mobility(self, r_ij):
        edge_edge_ij = r_ij - 2 * self.params.raft_radius
        if edge_edge_ij >= self.params.lubrication_threshold:
            return 1 / (
                8 * np.pi * self.params.dynamic_viscosity * self.params.raft_radius**3
            )
        else:
            x = max([0, edge_edge_ij / self.params.raft_radius])
            return self.calc_lubG(x) / self.params.dynamic_viscosity

    def _calc_trans_mobility(self, r_ij, lub_option):
        edge_edge_ij = r_ij - 2 * self.params.raft_radius
        if edge_edge_ij > self.params.lubrication_threshold:
            return 1 / (
                6 * np.pi * self.params.dynamic_viscosity * self.params.raft_radius
            )
        else:
            x = max([0, edge_edge_ij / self.params.raft_radius])
            if lub_option == "A":
                lub = self.calc_lubA(x)
            elif lub_option == "B":
                lub = self.calc_lubB(x)
            elif lub_option == "C":
                lub = self.calc_lubC(x)
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
            edge_edge_dists = []
            for j in range(i, n_rafts):
                r_ij_vec = state[j, :2] - state[i, :2]
                r_ij_norm = np.linalg.norm(r_ij_vec)
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

                edge_edge_dist = r_ij_norm - 2 * self.params.raft_radius
                edge_edge_dists.append(edge_edge_dists)
                phi_ij = phi_i
                torque_capill = self.capillary_torque[
                    int(edge_edge_dist + 0.5), int(phi_ij + 0.5)
                ]
                omegas[i] += rot_mob * torque_capill
                omegas[j] += -rot_mob * torque_capill
            omegas[i] += (
                self._calc_rot_mobility(min(edge_edge_dists)) * b_field_torque[i]
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
                self._calc_trans_mobility(2 * self.params.lubrication_threshold)
                * self.params.fluid_density
                * omegas[i] ** 2
                * self.params.raft_radius**7
            )
            vels[i, :] += prefactor * (1 / dist_bot_left**3 - 1 / dist_top_right**3)

        # pair interactions
        for i in range(n_rafts):
            for j in range(i, n_rafts):
                r_ij_vec = state[j, :2] - state[i, :2]
                r_ij_norm = np.linalg.norm(r_ij_vec)
                r_ij_angle = np.arctan2(r_ij_vec[1], r_ij_vec[0])
                r_ij_unit_vec = r_ij_vec / r_ij_norm
                r_ij_cross_z = np.array([r_ij_unit_vec[1], -r_ij_unit_vec[0]])
                edge_edge_dist = r_ij_norm - 2 * self.params.raft_radius
                phi_i = state[i, 2] - r_ij_angle
                phi_ij = phi_i

                f_mag_on_prefactor = (
                    3
                    * self.params.magnetic_constant
                    * mag_moments[i]
                    * mag_moments[j]
                    / (4 * np.pi)
                )
                f_mag_off_prefactor = f_mag_on_prefactor
                f_mag_on = self.calc_f_mag_on(r_ij_norm, phi_ij, f_mag_on_prefactor)
                f_mag_off = self.calc_f_mag_off(r_ij_norm, phi_ij, f_mag_off_prefactor)

                f_capillary = self.capillary_force[
                    int(edge_edge_dist + 0.5), int(phi_ij + 0.5)
                ]

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
                        -self._calc_trans_mobility(
                            2 * self.params.lubrication_threshold, "none"
                        )
                        * self.params.raft_repulsion_strength
                        * edge_edge_dist
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

                vels[i, :] += vel_ij_along_r_ij * r_ij_vec
                vels[i, :] += vel_ij_along_r_ij_cross_z * r_ij_cross_z
                vels[j, :] += -vel_ij_along_r_ij * r_ij_vec
                vels[j, :] += -vel_ij_along_r_ij_cross_z * r_ij_cross_z

    def get_rhs_new(self, t, state_flat):
        state = state_flat.reshape((-1, 3))

        b_field = self.bfield_calculator.calc_B_field(t)

        omega = self._get_omega(state, b_field)
        vel = self._get_vel(state, omega, b_field)

        state_derivative = np.stack([vel, omega[:, np.newaxis]], axis=1)
        return state_derivative.flatten()

    def integrate(self, n_slices: int):
        def rhs(t, state):
            # right_hand_side has self as the first argument so we write this
            # small wrapper to remove it from the call signature
            return self.get_rhs_new(t, state)

        scipy.solve_ivp(rhs)

    def calc_lubG(self, x):
        if x == 0:
            return 0.0212758 / self.params.raft_radius**3
        else:
            logx = np.log(x)
            return ((0.0212758 * (-logx) + 0.181089) * (-logx) + 0.381213) / (
                self.params.raft_radius**3 * ((-logx) * ((-logx) + 6.0425) + 6.32549)
            )

    def calc_lubA(self, x):
        if x == 0:
            return 0
        else:
            return (
                x
                * (-0.285524 * x + 0.095493 * x * np.log(x) + 0.106103)
                / self.params.raft_radius
            )  # unit: 1/um

    def calc_lubB(self, x):
        if x == 0.0:
            return 0.0212764 / self.params.raft_radius
        else:
            logx = np.log(x)
            return ((0.0212764 * (-logx) + 0.157378) * (-logx) + 0.269886) / (
                self.params.raft_radius * (-logx) * ((-logx) + 6.0425) + 6.32549
            )  # unit: 1/um

    def calc_lubC(self, x):
        return -self.params.raft_radius * self.calc_lubG(
            x
        )  # Gaurav: in the paper that is not true (eq 13, 14)

    def calc_f_mag_on(self, r_ij, phi_ij, prefactor):
        # eq 28
        return prefactor / r_ij**4 * (1 - 3 * np.cos(phi_ij) ** 2)

    def calc_f_mag_off(self, r_ij, phi_ij, prefactor):
        # eq 29
        return prefactor / r_ij**4 * 2 * np.cos(phi_ij) * np.sin(phi_ij)
