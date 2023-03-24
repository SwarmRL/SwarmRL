"""
Class for the genertic algorithm training routine.
"""
import os
from copy import deepcopy
from pathlib import Path

import jax.numpy as np
import numpy as onp
from dask.distributed import Client, wait

from swarmrl.gyms.gym import Gym


class GeneticTraining:
    """
    Class for the genetic training routine.
    """

    def __init__(
        self,
        gym: Gym,
        simulation_runner_generator: callable,
        n_episodes: int = 100,
        episode_length: int = 20,
        number_of_generations: int = 10,
        population_size: int = 10,
        number_of_parents: int = 2,
        number_of_simulations: int = 10,
        parent_selection_method: str = "sum",
        output_directory: str = ".",
        routine_name: str = "genetic_algorithm",
    ):
        """
        Constructor for the genetic training routine.

        Parameters
        ----------
        gym: Gym
            Gym to use for the training.
        simulation_runner_generator : callable
            A function that will returimport numpy as onp
            esults to.
        routine_name : str (default: "genetic_algorithm")
            Name of the training routine.

        Notes
        -----
        Currently the client is fixed to run on the local machine. This will be
        changed to a parameter in the future. The problem lies in espresso not being
        able to handle multiple threads and us not being able to force Dask to refresh
        a worker after each training is finished.
        """
        self.gym = gym
        self.simulation_runner_generator = simulation_runner_generator
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.number_of_generations = number_of_generations
        self.population_size = population_size
        self.number_of_parents = number_of_parents
        self.number_of_simulations = number_of_simulations
        self.output_directory = Path(f"{output_directory}/{routine_name}")

        # TODO: Make this a parameter
        self.client = Client(
            processes=False, threads_per_worker=5, n_workers=number_of_simulations
        )
        self.identifiers = range(population_size)

        lazy_splits = np.array_split(np.ones(population_size), number_of_parents)
        self.split_lengths = [len(split) for split in lazy_splits]

        # set the select function
        if parent_selection_method == "sum":
            self._select_fn = onp.sum
        elif parent_selection_method == "mean":
            self._select_fn = onp.mean
        elif parent_selection_method == "max":
            self._select_fn = onp.max

        # Create the output directory
        os.mkdir(Path(self.output_directory))

    @staticmethod
    def _train_network(
        name: Path,
        load_directory: str = None,
        gym: Gym = None,
        runner_generator: callable = None,
        select_fn: callable = None,
    ) -> tuple:
        """
        Train the network.

        Parameters
        ----------
        name : Path
            Name of the network and where to save the data.
        load_directory : str (default: None)
            Directory to load the model from. If None, a new model will be created.
        self.loss_fn = loss

        Returns
        -------
        reduced_rewards : float
            The reduced rewards of the agent.
        model_id : str
            The id of the model.
        """
        model_id = name.as_posix().split("_")[-1]
        os.makedirs(name)
        os.chdir(name)

        system_runner = runner_generator()  # get the runner

        if load_directory is None:
            gym.initialize_models()
        else:
            gym.restore_models(load_directory)

        rewards = gym.perform_rl_training(system_runner)
        gym.export_models()

        return (select_fn(rewards), model_id)

    def _get_gym(self):
        """
        Helper function to get the gym.
        """
        return deepcopy(self.gym)

    def _run_generation(
        self, generation: int, seed: bool = False, parent_ids: list = None
    ):
        """
        Run a generation of the training.

        Parameters
        ----------
        generation : int
            The number of the generation to run.
        seed : bool (default: False)
            Whether to seed the generation or not.
        parent_ids : list (default: None)
            The ids of the parents to use for the generation. If None, it should
            be seeded.
        """
        # Create the children directories
        children_names = [
            self.output_directory / f"_generation_{generation}" / f"_child_{i}"
            for i in self.identifiers
        ]
        print(children_names)
        # deploy the jobs
        if seed:
            generation_outputs = self.client.map(
                self._train_network,
                children_names,
                [None] * self.population_size,
                [deepcopy(self.gym)] * self.population_size,
                [self.simulation_runner_generator] * self.population_size,
                [self._select_fn] * self.population_size,
            )

        else:
            # get load paths for each parent
            load_paths = []
            for i, index in enumerate(parent_ids):
                load_paths += [
                    self.output_directory
                    / f"_generation_{generation - 1}"
                    / f"_child_{index}"
                ] * self.split_lengths[i]

            generation_outputs = self.client.map(
                self._train_network,
                children_names,
                [deepcopy(self.gym)] * self.population_size,
                load_paths,
                [deepcopy(self.gym)] * self.population_size,
                [self.simulation_runner_generator] * self.population_size,
                [self._select_fn] * self.population_size,
            )

        # Wait for results and load them from devices.
        wait(generation_outputs)
        generation_outputs = self.client.gather(generation_outputs)

        # Restart and wait for workers
        self.client.restart(wait_for_workers=False)
        self.client.wait_for_workers(self.number_of_simulations)

        return generation_outputs

    def _select_parents(self, generation_outputs: np.ndarray) -> list:
        """
        Select the parents for the next generation.

        Parameters
        ----------
        generation_outputs : np.ndarray (n_individuals, )
            The outputs of the generation.

        Returns
        -------
        ids : list
            The ids of the parents.
        """
        rewards = [item[0] for item in generation_outputs]
        ids = [item[1] for item in generation_outputs]

        # First get best parent
        max_reward_index = np.argmax(rewards)
        chosen_id = ids[max_reward_index]

        # Pick mutations
        if self.number_of_parents == 1:
            return [chosen_id]
        else:
            random_ids = onp.random.choice(
                ids, size=self.number_of_parents - 1, replace=False
            )
            return [chosen_id] + list(random_ids)

    def train_model(self):
        """
        Train the model.
        """
        generation = 0
        # Seed genetic process
        seed_outputs = self._run_generation(generation=generation, seed=True)
        parents = self._select_parents(seed_outputs)

        # Loop over generations
        for _ in range(self.number_of_generations - 1):
            generation += 1  # Update the generation
            generation_outputs = self._run_generation(
                generation=generation, parent_ids=parents
            )
            parents = self._select_parents(generation_outputs)

        return self.output_directory / f"_generation_{generation}" / "_child_0"
