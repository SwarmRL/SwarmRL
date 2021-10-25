"""
Parent class for the engine.
"""


class Engine:
    """
    Parent class for an engine.

    An engine is an object that can generate data for the environment. Currently we
    have only an espresso model but this should be kept generic to allow for an
    experimental interface.
    """
    def __init__(self):
        """
        Constructor for the engine.
        """
        pass

    def run(self):
        """
        Begin generating data.

        Returns
        -------

        """
        raise NotImplementedError("Implemented in child class.")
