"""
Parent class for the engine.
"""

from swarmrl.force_functions import ForceFunction


class Engine:
    """
    Parent class for an engine.

    An engine is an object that can generate data for the environment. Currently we
    have only an espresso model but this should be kept generic to allow for an
    experimental interface.
    """

    def integrate(
        self,
        n_slices: int,
        force_model: ForceFunction,
    ) -> None:
        """

        Parameters
        ----------
        n_slices: int
            Number of time slices to integrate
        force_model
            A an instance of ForceFunction
        """
        raise NotImplementedError

    def get_particle_data(self) -> dict:
        """
        Get type, id, position, velocity and director of the particles
        as a dict of np.array
        """
        raise NotImplementedError

    def finalize(self):
        """
        Optional: to clean up after finishing the simulation (e.g. writing the last
        chunks of trajectory)
        """
        pass
