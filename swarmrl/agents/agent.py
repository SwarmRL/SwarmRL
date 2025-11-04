"""
Parent class for all agents
"""

import typing

from swarmrl.actions.actions import Action
from swarmrl.components.colloid import Colloid


class Agent:
    """
    Parent class for a SwarmRL Agent.
    """

    _killed = False

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
        raise NotImplementedError("Implemented in Child class.")
