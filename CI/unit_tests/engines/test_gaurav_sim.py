import pathlib
import tempfile
import unittest as ut

import h5py
import numpy as np
import pint
import tqdm

import swarmrl.engine.gaurav_sim as gaurav_sim


class GauravSimTest(ut.TestCase):
    def test_setup(self):
        ureg = pint.UnitRegistry()
        Q_ = ureg.Quantity
        params = gaurav_sim.GauravSimParams(
            ureg=ureg,
            box_length=Q_(10000, "micrometer"),
            time_step=Q_(1e-3, "second"),
            time_slice=Q_(1, "second"),
            snapshot_interval=Q_(0.002, "second"),
            raft_radius=Q_(150, "micrometer"),
            raft_repulsion_strength=Q_(1e-7, "newton"),
            dynamic_viscosity=Q_(1e-3, "Pa * s"),
            fluid_density=Q_(1000, "kg / m**3"),
            lubrication_threshold=Q_(15, "micrometer"),
            magnetic_constant=Q_(4 * np.pi * 1e-7, "newton /ampere**2"),
            capillary_force_data_path=pathlib.Path(
                "/work/clohrmann/mpi_collab/capillaryForceAndTorque_sym6"
            ),  # TODO
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            sim = gaurav_sim.GauravSim(params, temp_dir, with_precalc_capillary=True)
            mag_mom = Q_(1e-8, "ampere * meter**2")
            sim.add_raft(Q_(np.array([300, 500]), "micrometer"), 0, mag_mom)
            sim.add_raft(Q_(np.array([1000, 500]), "micrometer"), np.pi / 2, mag_mom)
            sim.add_raft(
                Q_(np.array([700, 1000]), "micrometer"), np.pi / 2 - 0.1, mag_mom
            )

            action = gaurav_sim.GauravAction(
                Q_(np.array(2 * [10]), "mT").m_as("sim_magnetic_field"),
                Q_(np.array(2 * [10]), "hertz").m_as("1/sim_time"),
                np.array([0, np.pi / 2]),  # radian
                Q_(np.array([0, 0]), "mT").m_as("sim_magnetic_field"),
            )
            model = gaurav_sim.ConstantAction(action)

            for _ in tqdm.tqdm(range(10)):
                sim.integrate(1, model)

            with h5py.File(sim.h5_filename) as h5_file:
                group = h5_file["rafts"]
                poss = np.array(group["Unwrapped_Positions"])
                angles = np.array(group["Alphas"])[:, :, 0]
                times = np.array(group["Times"])[:, 0, 0]
                assert len(poss) > 0
                assert len(angles) == len(times)
                # TODO: write actual test


if __name__ == "__main__":
    ut.main()
