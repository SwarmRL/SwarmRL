"""
Child class for the ResoBee engine
"""
from swarmrl.engine.engine import Engine
from swarmrl.models.interaction_model import InteractionModel


class ResoBee(Engine):
    """
    Child class for the ResoBee Engine.
    """

    def integrate(
        self,
        n_slices: int,
        force_model: InteractionModel,
    ) -> None:
        """

        Parameters
        ----------
        n_slices: int
            Number of time slices to integrate
        force_model
            A an instance of swarmrl.models.interaction_model.InteractionModel
        """
        raise NotImplementedError

    def get_particle_data(self) -> dict:
        """
        Get type, id, position, velocity and director of the particles
        as a dict of np.array
        """
        raise NotImplementedError
