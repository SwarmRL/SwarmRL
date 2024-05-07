import typing

import h5py as hf
import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.classical_agent import ClassicalAgent


class AgentFromTrajectory(ClassicalAgent):
    """
    Class for agents that follow a specific trajectory
    """

    def __init__(
        self,
        trajectory: str or np.array = None,
        force_function: callable = None,
        time_slice: float = 0.01,
        gammas: typing.List[float] = None,
        acts_on_types: typing.List[float] = [1],
        params: np.array = None,
        home_pos: np.array = np.array([0, 0, 0]),
    ):
        """
        Initialize the AgentFromTrajectory object.

        Parameters
        ----------
        trajectory_file : str or list
            str: Path to the trajectory file. If provided, the file will be loaded.
            list: List of wanted positions. If provided, the list will be used.
        force_function : callable
            Function that describes the movement of the agent.
            If provided, it will be used instead of the trajectory file.
        time_slice : float, optional
            Time slice for the movement. Default is 0.01.
        gammas : list
            List of gamma values for the trajectory.
            gammas[0] = translational gamma, gammas[1] = rotational gamma.
        acts_on_types : int, optional (default: 1)
            Colloid types that are affected by the agent.
        params : np.array, optional
            Parameters for the trajectory function.
            If trajectory_file is not None, params is ignored.
        home_pos : np.array, optional (default: [0, 0, 0])
            Home position of the agent.

        Raises
        ------
        ValueError
            If both/neither of trajectory_file and force_function are provided.
        """
        if trajectory is not None and force_function is None:
            if isinstance(trajectory, str):
                self.wanted_pos = self.load_trajectory(trajectory)
            if isinstance(trajectory, (list, np.ndarray)):
                self.wanted_pos = trajectory
            self.force_function = None
        elif force_function is not None and trajectory is None:
            self.force_function = force_function
        else:
            raise ValueError(
                "Provide either a trajectory file or a force function, "
                "not both or neither."
            )
        self.acts_on_types = acts_on_types
        self.home_pos = home_pos

        self.params = params
        self.t = 0
        self.index_tracker = -1
        self.time_slice = time_slice

    def load_trajectory(self, trajectory_file: str):
        """
        Load the trajectory from a file.
        """
        db = hf.File(f"{trajectory_file}/trajectory.hdf5")
        return db["Wanted_Positions"][:]

    def update_force_function(self, force_function: callable):
        """
        Change the force function.
        """
        self.force_function = force_function

    def calc_force_next_pos(self, pos, next_pos, velocity, time_slice):
        """
        Calculate the force needed to reach next_pos in time_slice.
        """
        mass = 1
        if velocity is None:
            velocity = np.array([0, 0, 0])
        return (next_pos - pos - velocity * time_slice) * 2 * mass / time_slice**2

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        self.index_tracker += 1
        self.t += self.time_slice

        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            if self.force_function is not None:
                force = self.force_function(
                    self.t, colloid.pos, colloid.director, self.home_pos, self.params
                )
                force_value = np.linalg.norm(force)
                new_direction = force / force_value
                actions.append(Action(force=force_value, new_direction=new_direction))

            else:
                pos = self.wanted_pos[self.index_tracker]
                next_pos = self.wanted_pos[self.index_tracker + 1]
                force = self.calc_force_next_pos(
                    pos, next_pos, colloid.velocity, self.time_slice
                )
                force_value = np.linalg.norm(force)
                new_direction = force / force_value
                actions.append(Action(force=force_value, new_direction=new_direction))

        return actions


def harmonic_1d(t, pos, director, home_pos, params):
    """
    harmonic motion along x-axis
    ----------------------------
    params[0]: amplitude
    params[1]: frequency
    params[2]: offset in y-direction
    """
    force_x = params[0] * np.cos(params[1] * t)
    force_y = home_pos[1] - pos[1]
    force_z = 0
    return force_x, force_y, force_z


def harmonic_2d(t, pos, director, home_pos, params):
    """
    harmonic motion in x-y-plane
    -----------------------------
    params[0]: amplitude
    params[1]: frequency
    """
    force_x = params[0] * np.cos(params[1] * t)
    force_y = params[0] * np.sin(params[1] * t)
    force_z = 0
    return force_x, force_y, force_z


def no_force(t, pos, director, home_pos, params):
    """
    no force
    """
    return 0, 0, 0
