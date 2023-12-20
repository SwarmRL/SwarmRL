"""
Parent class for all agents
"""

import typing

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid


class TrainableAgent(Agent):
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
                Return the action the colloid should take.
        kill_switch : bool
                Flag capable of ending simulation.
        """
        raise NotImplementedError("Implemented in Child class.")

    def update_agent(self) -> tuple:
        """
        Updates the network of a trainable agent.

        Returns
        -------
        rewards : float
                Net reward for the agent.
        killed : bool
                Whether or not this agent killed the
                simulation.
        """
        raise NotImplementedError("Implemented in Child class.")
