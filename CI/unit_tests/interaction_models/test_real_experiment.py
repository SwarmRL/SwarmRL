import unittest as ut

import swarmrl.engine.real_experiment
import swarmrl.models.dummy_models


class TestRealExperiment(ut.TestCase):
    def test_runs(self):
        # TODO write actual test
        pass
        """
        runner = swarmrl.engine.real_experiment.RealExperiment()
        runner.setup_simulation()
        f_model = swarmrl.models.dummy_models.ConstForce(123)
        runner.integrate(100, f_model)
        runner.finalize()
        """


if __name__ == "__main__":
    ut.main()
