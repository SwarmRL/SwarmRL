import tempfile
import unittest as ut

import h5py
import numpy as np

import swarmrl.engine.gaurav_sim as gaurav_sim


class GauravSimTest(ut.TestCase):
    def test_setup(self):
        params = gaurav_sim.GauravSimParams(
            box_length=1000,
            time_step=0.01,
            time_slice=1,
            snapshot_interval=0.25,
            raft_radius=1,
            raft_repulsion_strength=0.1,
            dynamic_viscosity=0.01,
            fluid_density=1,
            lubrication_threshold=1.01,
            magnetic_constant=0.01,
            capillary_force_data_path=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            sim = gaurav_sim.GauravSim(params, temp_dir, with_precalc_capillary=False)
            mag_mom = 0.1
            sim.add_raft(np.array([50, 50]), np.pi, mag_mom)
            sim.add_raft(np.array([55, 50]), np.pi, mag_mom)
            sim.add_raft(np.array([60, 50]), np.pi, mag_mom)

            action = gaurav_sim.GauravAction(
                np.array([1, 1]),
                np.array([0.1, 0.1]),
                np.array([0, 0]),
                np.array([0, 0]),
            )
            model = gaurav_sim.ConstantAction(action)

            sim.integrate(10, model)

            with h5py.File(sim.h5_filename) as h5_file:
                group = h5_file["rafts"]
                poss = group["Unwrapped_Positions"]
                assert len(poss) > 0


if __name__ == "__main__":
    ut.main()
