"""
Class for the genertic algorithm training routine.
"""
import os
from pathlib import Path

from dask.distributed import Client

from swarmrl.rl_protocols.rl_protocol import RLProtocol


class GeneticTraining:
    """
    Class for the genetic training routine.
    """

    def __init__(
        self,
        base_protocol: RLProtocol,
        simulation_runner_generator: callable,
        number_of_generations: int = 10,
        population_size: int = 10,
        number_of_parents: int = 2,
        number_of_simulations: int = 10,
        output_directory: str = ".",
        routine_name: str = "genetic_algorithm",
    ):
        """
        Constructor for the genetic training routine.

        Parameters
        ----------
        base_protocol : RLProtocol
            The base protocol to use for the training.
        simulation_runner_generator : callable
            A function that will return a system runner
        number_of_generations : int (default: 10)
            Number of generations to train for.
        population_size : int (default: 10)
            Number of individuals in the population.
        number_of_parents : int (default: 2)
            Number of parents to select for each child.
        number_of_simulations : int (default: 10)
            Number of simulations to run in parallel.
        output_directory : str (default: ".")
            Directory to save the results to.
        routine_name : str (default: "genetic_algorithm")
            Name of the training routine.

        Notes
        -----
        Currently the client is fixed to run on the local machine. This will be
        changed to a parameter in the future. The problem lies in espresso not being
        able to handle multiple threads and us not being able to force Dask to refresh
        a worker after each training is finished.
        """
        self.base_protocol = base_protocol
        self.simulation_runner_generator = simulation_runner_generator
        self.number_of_generations = number_of_generations
        self.population_size = population_size
        self.number_of_parents = number_of_parents
        self.number_of_simulations = number_of_simulations
        self.output_directory = Path(f"{output_directory}/{routine_name}")

        # TODO: Make this a parameter
        self.client = Client(
            processes=False, threads_per_worker=5, n_workers=number_of_simulations
        )

        os.mkdir(Path(self.output_directory))
