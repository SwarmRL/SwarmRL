"""
Class for submitting many jobs in parallel to a cluster.
"""
import logging
import os
import webbrowser
from pathlib import Path
from typing import List

from dask.distributed import Client, LocalCluster, wait
from dask_jobqueue import JobQueueCluster

from swarmrl.gyms.gym import Gym


class EnsembleTraining:
    """
    Class for ensemble training.
    """

    def __init__(
        self,
        gym: Gym,
        simulation_runner_generator: callable,
        number_of_ensembles: int,
        episode_length: int,
        n_episodes: int,
        n_parallel_jobs: int = None,
        load_path: Path = None,
        cluster: JobQueueCluster = None,
        output_dir: Path = Path("./ensembled-training"),
    ) -> None:
        """
        Constructor for the ensemble training routine.

        Parameters
        ----------
        gym : Gym
            The gym to train.
        number_of_ensmbles : int
            The number of ensembles to train.
        episode_length : int
            The length of each episode.
        n_episodes : int
            The number of episodes to train for.
        simulation_runner_generator : callable
            A callable function that returns a simulation runner.
        n_parallel_jobs : int
            The number of parallel jobs to run.
        cluster : JobQueueCluster
            The cluster to run the jobs on.
            If None, the jobs will be run locally.
        load_path : Path or str or None (default)
            The path to load the models from.
        output_dir : Path or str or None (default)
            The directory to save the models to.

        """
        self.simulation_runner_generator = simulation_runner_generator
        self.output_dir = output_dir
        self.load_path = load_path
        self.episode_length = episode_length
        self.n_episodes = n_episodes

        # Update the default parameters.
        if n_parallel_jobs is None:
            n_parallel_jobs = number_of_ensembles

        self.gym = gym
        self.number_of_ensembles = number_of_ensembles
        self.n_parallel_jobs = n_parallel_jobs

        # Use default local cluster if None is given.
        if cluster is None:
            cluster = LocalCluster(
                processes=True, threads_per_worker=10, silence_logs=logging.ERROR
            )
        self.cluster = cluster
        self.client = Client(cluster)

        self.cluster.scale(n=self.n_parallel_jobs)
        webbrowser.open(self.client.dashboard_link)

        # Create the output directory if needed.
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

    @staticmethod
    def _train_model(
        save_path: str,
        gym: Gym,
        system_runner: callable,
        load_directory: str = None,
        episode_length: int = 100,
        n_episodes: int = 100,
    ) -> List:
        """
        Job to submit to dask.

        Parameters
        ----------
        ensemble_id : int
            The ensemble id.
        gym : Gym
            The gym to train.
        load_directory : str
            The directory to load the models from.
        episode_length : int
            The length of each episode.
        n_episodes : int
            The number of episodes to train for.
        """
        model_id = save_path.split("_")[-1]
        # Create the new paths.
        os.makedirs(save_path)
        os.chdir(save_path)

        # Get the system runner.
        system_runner = system_runner()
        if load_directory is not None:
            gym.restore_models(directory=load_directory)
        else:
            gym.initialize_models()

        # Train the gym.
        rewards = gym.perform_rl_training(
            system_runner,
            n_episodes=n_episodes,
            episode_length=episode_length,
            initialize=True,
            load_bar=False,
        )
        gym.export_models()

        return rewards, model_id

    def train_ensemble(self) -> None:
        """
        Train the ensemble.

        Returns
        -------
        model_performance : dict
            A dictionary of the model performance.
            structure of the dictionary is: {model_id: rewards}.
        """
        futures = []
        names = [
            (self.output_dir / f"ensemble_{i}").resolve().as_posix()
            for i in range(self.number_of_ensembles)
        ]

        for i in range(self.number_of_ensembles // self.n_parallel_jobs):
            block = self.client.map(
                self._train_model,
                names[i * self.n_parallel_jobs : (i + 1) * self.n_parallel_jobs],
                [self.gym] * self.n_parallel_jobs,
                [self.simulation_runner_generator] * self.n_parallel_jobs,
                [self.load_path] * self.n_parallel_jobs,
                [self.episode_length] * self.n_parallel_jobs,
                [self.n_episodes] * self.n_parallel_jobs,
            )
        _ = wait(block)
        futures += self.client.gather(block)
        _ = self.client.restart(wait_for_workers=False)
        _ = self.client.wait_for_workers(self.n_parallel_jobs)

        # shut down the cluster
        self.cluster.close()
        self.client.close()

        return {model_id: rewards for rewards, model_id in futures}
