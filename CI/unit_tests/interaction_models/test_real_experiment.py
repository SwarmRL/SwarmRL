"""
Module for real experiment unit tests.
"""
import unittest as ut


class DummyConnection:
    """
    Dummy connect class for the test.
    """

    def recv(self, bytes: int):
        """
        Dummy receive method.
        """
        return [1, 2, 3, 4, 5]

    def sendall(self):
        """
        Dummy sendall method.
        """
        pass

    def close(self):
        """
        Dummy close method.
        """
        pass


class TestRealExperiment(ut.TestCase):
    def test_runs(self):
        """
        Test the real experiment interface.
        """
        # runner = swarmrl.engine.real_experiment.RealExperiment(DummyConnection())
        # runner.setup_simulation()
        # f_model = swarmrl.models.dummy_models.ConstForce(123)
        # runner.integrate(100, f_model)
        # runner.finalize()


if __name__ == "__main__":
    ut.main()
