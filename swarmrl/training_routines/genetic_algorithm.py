"""
Class for the genertic algorithm training routine.
"""
import logging
import os
import webbrowser
from copy import deepcopy
from pathlib import Path
from typing import List

import jax.numpy as np
import numpy as onp
from dask.distributed import Client, LocalCluster, wait
from dask_jobqueue import JobQueueCluster
from rich import print
from rich.progress import BarColumn, Progress, TimeRemainingColumn

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
        parent_selection_method: str = "sum",
        output_directory: str = ".",
        routine_name: str = "genetic_algorithm",
        parallel_jobs: int = None,
        cluster: JobQueueCluster = None,
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
        n_episodes : int
            Number of episodes in each lifespan
        number_of_generations : int (default: 10)
            Number of generations to run
        population_size : int (default: 10)
            Number of individuals in the population
        number_of_parents : int (default: 2)
            Number of parents to select for each generation
        parent_selection_method : str
            How to reduce the life reward. Either sum or mean.
        output_directory : str (default: ".")
            Output directory of the run.
        routine_name : str (default: "genetic_algorithm")
            Name of the training routine.
        cluster : JobQueueCluster
            The cluster to run the jobs on.
            If None, the jobs will be run locally.
        parallel_jobs : int
            The number of parallel jobs to run.
            If None, the population size is used.

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
        self.output_directory = Path(f"{output_directory}/{routine_name}")

        if parallel_jobs is None:
            parallel_jobs = population_size
        self.parallel_jobs = parallel_jobs

        # Use default local cluster if None is given.
        if cluster is None:
            cluster = LocalCluster(
                processes=True, threads_per_worker=5, silence_logs=logging.ERROR
            )
        self.cluster = cluster

        self.client = Client(cluster)

        self.cluster.scale(n=self.parallel_jobs)
        webbrowser.open(self.client.dashboard_link)
        # Decide on parent splits
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
        episode_length: int = None,
        n_episodes: int = None,
    ) -> tuple:
        """
        Train the network.

        Parameters
        ----------
        name : Path
            Name of the network and where to save the data.
        load_directory : str (default: None)
            Directory to load the model from. If None, a new model will be created.
        gym : Gym
                Gym to use for training.
        runner_generator : callable
                Function that returns a system_runner.
        select_fn : callable
                Function for reducing training rewards to a single
                number.
        episode_length : int
                Length of one episode.
        n_episodes : int
                Number of episodes in the lifespan of the child.

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

        rewards = gym.perform_rl_training(
            system_runner,
            episode_length=episode_length,
            n_episodes=n_episodes,
            initialize=True,
            load_bar=False,
        )
        gym.export_models()

        return (select_fn(rewards), model_id)

    def _deploy_jobs(
        self,
        child_names: List[Path],
        load_paths: List[Path],
    ) -> List[float]:
        """
        Function to send jobs to the cluster.

        Parameters
        ----------
        child_names : List[Path]
            List of paths to save the models to.
        load_paths : List[Path, None]
            List of paths to load the models from.

        Returns
        -------
        Returns the outcome of the job deployment.
        """
        futures = []

        for i in range(self.population_size // self.parallel_jobs):
            block = self.client.map(
                self._train_network,
                child_names[i * self.parallel_jobs : (i + 1) * self.parallel_jobs],
                load_paths[i * self.parallel_jobs : (i + 1) * self.parallel_jobs],
                [deepcopy(self.gym)] * self.parallel_jobs,
                [self.simulation_runner_generator] * self.parallel_jobs,
                [self._select_fn] * self.parallel_jobs,
                [self.episode_length] * self.parallel_jobs,
                [self.n_episodes] * self.parallel_jobs,
            )
            _ = wait(block)
            futures += self.client.gather(block)
            # Restart and wait for workers
            _ = self.client.restart(wait_for_workers=False)
            _ = self.client.wait_for_workers(self.parallel_jobs)

        return futures

    def _run_generation(
        self, generation: int, seed: bool = False, parent_ids: list = None
    ) -> List:
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

        Returns
        -------
        generation_outputs : list
                A list of rewards values form which parents will be chosen.
        """
        # Create the children directories
        children_names = [
            (
                self.output_directory / f"_generation_{generation}" / f"_child_{i}"
            ).resolve()
            for i in self.identifiers
        ]
        # deploy the jobs
        if seed:
            generation_outputs = self._deploy_jobs(
                children_names, [None] * self.population_size
            )

        else:
            # get load paths for each parent
            load_paths = []
            for i, index in enumerate(parent_ids):
                load_paths += [
                    self.output_directory
                    / f"_generation_{generation - 1}"
                    / f"_child_{index}"
                    / "Models"
                ] * self.split_lengths[i]

            load_paths = [item.resolve().as_posix() for item in load_paths]

            generation_outputs = self._deploy_jobs(children_names, load_paths)

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
        chosen_reward: float
            Reward of the chosen child.
        """
        rewards = [item[0] for item in generation_outputs]
        ids = [item[1] for item in generation_outputs]

        # First get best parent
        max_reward_index = np.argmax(np.array(rewards))
        chosen_id = ids[max_reward_index]

        # Pick mutations
        if self.number_of_parents == 1:
            return [chosen_id], rewards[max_reward_index]
        else:
            random_ids = onp.random.choice(
                ids, size=self.number_of_parents - 1, replace=False
            )
            return [chosen_id] + list(random_ids), rewards[max_reward_index]

    def train_model(self):
        """
        Train the model.
        """
        generation = 0
        # Seed genetic process
        seed_outputs = self._run_generation(generation=generation, seed=True)
        parents, reward = self._select_parents(seed_outputs)

        # Loop over generations

        progress = Progress(
            "Generation: {task.fields[generation]}",
            BarColumn(),
            "Best generation reward: {task.fields[best_reward]:.2f} ",
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task(
                "Genetic training",
                total=self.number_of_generations - 1,
                generation=generation,
                best_reward=reward,
            )
            for _ in range(self.number_of_generations - 1):
                generation += 1  # Update the generation
                generation_outputs = self._run_generation(
                    generation=generation, parent_ids=parents
                )
                parents, reward = self._select_parents(generation_outputs)
                progress.update(
                    task,
                    advance=1,
                    generation=generation,
                    best_reward=reward,
                )

        best_model_path = (
            self.output_directory / f"_generation_{generation}" / f"_child_{parents[0]}"
        )
        print(f"Best Model: {best_model_path.as_posix()}")
        print(f"Best Reward: {reward:.2f}")

        return best_model_path
