import multiprocessing
import tempfile
import time
import traceback
import unittest as ut

import numpy as np
import pint

import swarmrl.engine.espresso as espresso
from swarmrl.agents import dummy_models
from swarmrl.force_functions import ForceFunction
from swarmrl.utils import utils


class Process(multiprocessing.Process):
    """
    Process class for use in multi simulation testing.
    """

    def __init__(self, *args, **kwargs):
        """
        Multiprocessing class constructor.
        """
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        """
        Run the process and catch exceptions.
        """
        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        """
        Exception property to be stored by the process.
        """
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class Simulation:
    def __init__(
        self,
        outfolder,
        seed,
        test_number,
    ):
        self.outfolder = outfolder
        self.seed = seed
        self.test_number = test_number

    def simulate_model(self):
        ureg = pint.UnitRegistry()

        if self.test_number == 0:
            time_step = ureg.Quantity(0.1, "second")
            time_slice = time_step * 5
            write_interval = time_step * 9

        if self.test_number == 1:
            time_step = ureg.Quantity(0.1, "second")
            time_slice = time_step * 7
            write_interval = time_step * 3

        if self.test_number == 2:
            time_step = ureg.Quantity(0.1, "second")
            time_slice = time_step * 2
            write_interval = time_step * 2

        md_params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(293, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(293, "kelvin"),
            box_length=ureg.Quantity(3 * [10], "micrometer"),
            time_step=time_step,
            time_slice=time_slice,
            write_interval=write_interval,
        )
        system_runner = espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=self.seed,
            out_folder=self.outfolder,
            write_chunk_size=10,
        )
        system_runner.add_colloids(
            1,
            ureg.Quantity(0.2, "micrometer"),
            ureg.Quantity(np.array([5, 5, 0]), "micrometer"),
            ureg.Quantity(1, "micrometer"),
            type_colloid=0,
        )
        force = 1
        agent = dummy_models.ConstForce(force)
        force_fn = ForceFunction(agents={"0": agent})

        if self.test_number == 0:
            np.testing.assert_equal(system_runner.system.time, 0)
            system_runner.integrate(2, force_fn)
            np.testing.assert_equal(system_runner.step_idx, 10)
            np.testing.assert_equal(system_runner.slice_idx, 2)
            # should be ceil(steps/steps_per_slice)
            np.testing.assert_equal(system_runner.write_idx, 2)
            # should be ceil(steps/steps_per_write_interval)
            np.testing.assert_almost_equal(system_runner.system.time, 1)
            np.testing.assert_equal(system_runner.params.steps_per_write_interval, 9)
            np.testing.assert_equal(system_runner.params.steps_per_slice, 5)
            np.testing.assert_equal(len(system_runner.traj_holder["Times"]), 2)
            system_runner.integrate(3, force_fn)
            np.testing.assert_equal(system_runner.step_idx, 25)
            np.testing.assert_equal(system_runner.slice_idx, 5)
            np.testing.assert_equal(system_runner.write_idx, 3)
            np.testing.assert_almost_equal(system_runner.system.time, 2.5)
            np.testing.assert_equal(len(system_runner.traj_holder["Times"]), 3)
            # Nothing is written to a file here because write_chunk_size is not reached

        if self.test_number == 1:
            np.testing.assert_equal(system_runner.system.time, 0)
            system_runner.integrate(4, force_fn)
            np.testing.assert_equal(system_runner.step_idx, 28)
            np.testing.assert_equal(system_runner.slice_idx, 4)
            np.testing.assert_equal(system_runner.write_idx, 10)
            np.testing.assert_almost_equal(system_runner.system.time, 2.8)
            np.testing.assert_equal(system_runner.params.steps_per_write_interval, 3)
            np.testing.assert_equal(system_runner.params.steps_per_slice, 7)
            np.testing.assert_equal(len(system_runner.traj_holder["Times"]), 0)
            # After write_chunk_size the holder is emptied
            system_runner.integrate(2, force_fn)
            np.testing.assert_equal(system_runner.step_idx, 42)
            np.testing.assert_equal(system_runner.slice_idx, 6)
            np.testing.assert_equal(system_runner.write_idx, 14)
            np.testing.assert_almost_equal(system_runner.system.time, 4.2)
            np.testing.assert_equal(len(system_runner.traj_holder["Times"]), 4)

        if self.test_number == 2:
            np.testing.assert_equal(system_runner.system.time, 0)
            system_runner.integrate(4, force_fn)
            np.testing.assert_equal(system_runner.step_idx, 8)
            np.testing.assert_equal(system_runner.slice_idx, 4)
            np.testing.assert_equal(system_runner.write_idx, 4)
            np.testing.assert_almost_equal(system_runner.system.time, 0.8)
            np.testing.assert_equal(system_runner.params.steps_per_write_interval, 2)
            np.testing.assert_equal(system_runner.params.steps_per_slice, 2)
            np.testing.assert_equal(len(system_runner.traj_holder["Times"]), 4)
            system_runner.integrate(2, force_fn)
            np.testing.assert_equal(system_runner.step_idx, 12)
            np.testing.assert_equal(system_runner.slice_idx, 6)
            np.testing.assert_equal(system_runner.write_idx, 6)
            np.testing.assert_almost_equal(system_runner.system.time, 1.2)
            np.testing.assert_equal(len(system_runner.traj_holder["Times"]), 6)
            # Nothing is written to a file here because write_chunk_size is not reached

        system_runner.finalize()


class TestSimulationIntegration(ut.TestCase):
    """
    Functional test, also to be used as an example for simulation scripts
    """

    # Parameters usually provided as commandline input
    loglevel_terminal = "info"
    seed = 42

    def test_0(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_0"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.test_number = 0
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.seed,
                self.test_number,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(5)
            process.join(1)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertIsNone(process.exception)

    def test_1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_1"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.test_number = 1
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.seed,
                self.test_number,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(5)
            process.join(1)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertIsNone(process.exception)

    def test_2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_2"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.test_number = 2
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.seed,
                self.test_number,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(5)
            process.join(1)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertIsNone(process.exception)


if __name__ == "__main__":
    ut.main()
