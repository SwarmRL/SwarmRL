"""
Parent class for all agents
"""

import typing

from loguru import logger

from swarmrl.actions.actions import Action
from swarmrl.components.colloid import Colloid


class Agent:
    """
    Parent class for a SwarmRL Agent.
    """

    _killed = False

    def on_rollout_end(self, *, terminated: bool, truncated: bool) -> typing.Any | None:
        """
        Handle the end of a rollout (trainer episode).

        Agents that learn from rollout data can override this hook to apply boundary
        semantics and update themselves. Non-learning agents return ``None``.
        """
        return None

    def persist_trajectory(self, trajectory) -> None:
        """
        Persist trajectory data if a storage backend is attached.
        """
        storage = getattr(self, "trajectory_storage", None)
        if storage is not None:
            storage.write(trajectory)

    def finalize(self) -> None:
        """Finalize trajectory storage if a backend is attached."""
        storage = getattr(self, "trajectory_storage", None)
        if storage is not None:
            storage.finalize()

    @property
    def kill_switch(self):
        """
        If true, kill the simulation.
        """
        return self._killed

    @kill_switch.setter
    def kill_switch(self, value):
        """
        Set the kill switch.
        """
        self._killed = value

    def calc_action(
        self, colloids: typing.List[Colloid]
    ) -> typing.Tuple[typing.List[Action]]:
        """
        Compute the state of the system based on the current colloid position.

        Returns
        -------
        actions: typing.List[Action]
                Return the action the colloid should take. Only return actions for the
                colloid types that the agent should act on.
        kill_switch : bool
                Flag capable of ending simulation.
        """
        raise NotImplementedError("Implemented in Child class.")

    def calc_reward(
        self, colloids: typing.List[Colloid], external_reward: float = 0.0
    ) -> None:
        """
        Compute the reward for the agent based on the current state.

        Parameters
        ----------
        colloids: typing.List[Colloid]
                List of colloids in the simulation.
        external_reward: float
                External reward from the environment.

        """
        logger.info(
            f"{self.__class__.__name__} does not implement calc_reward, skipping."
        )
