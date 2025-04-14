"""
Espresso interaction model capable of handling a neural network as a function.
"""

import typing

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.components.colloid import Colloid


class ForceFunction:
    """
    Class to bridge agents with an engine.
    """

    _kill_switch: bool = False

    def __init__(
        self,
        agents: dict,
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        agents : dict
            Agents used in the simulations.
        """
        super().__init__()
        self.agents = agents

        # Used in the data saving.
        self.particle_types = [type_ for type_ in self.agents]

    @property
    def kill_switch(self):
        """
        If true, kill the simulation.
        """
        return self._kill_switch

    @kill_switch.setter
    def kill_switch(self, value):
        """
        Set the kill switch.
        """
        self._kill_switch = value

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Compute the state of the system based on the current colloid position.

        In the case of the ML models, this method undertakes the following steps:

        1. Compute observable
        2. Compute action probabilities
        3. Compute action

        Returns
        -------
        action: Action
                Return the action the colloid should take.
        kill_switch : bool
                Flag capable of ending simulation.
        """
        # Prepare the data storage.
        actions = {int(np.copy(colloid.id)): Action() for colloid in colloids}
        switches = []

        # Loop over particle types and compute actions.
        for agent in self.agents:
            computed_actions = self.agents[agent].calc_action(colloids=colloids)
            switches.append(self.agents[agent].kill_switch)

            count = 0  # Count the colloids of a specific species.
            for colloid in colloids:
                if str(colloid.type) == agent:
                    actions[colloid.id] = computed_actions[count]
                    count += 1

        self.kill_switch = any(switches)

        return list(actions.values())

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

        for agent in self.agents:
            self.agents[agent].calc_reward(
                colloids=colloids, external_reward=external_reward
            )
