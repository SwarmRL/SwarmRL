"""
Module for the implementation of an environment.
"""
from swarmrl import InteractionModel


class Environment:
    """
    Class for the RL environment.

    The RL environment contains within it the RL model. It will run the simulation,
    compute the state of the simulation and then update the RL model.
    """
    def __init__(self, model: InteractionModel, engine: object):
        """
        Constructor for the Environment.

        Parameters
        ----------
        model : InteractionModel
                Model from which to compute the external forces.
        engine : object
                Object from which environment data should be generated.
        """
        self.model = model
        self.engine = engine

    def compute_state(self):
        """
        Compute the state of the system.

        The state of the system are the observables that are used to compute the reward
        for the learner.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")
