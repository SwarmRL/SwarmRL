"""
Test the mlp rl module.
"""
from swarmrl.gyms.mlp_rl import MLPRL


class TestMLPRL:
    """
    Test the MLP RL module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class.
        """
        cls.rl_trainer = MLPRL(
            actor=None,
            critic=None,
            task=None,
            loss=None,
            observable=None,
            n_particles=2,
        )
