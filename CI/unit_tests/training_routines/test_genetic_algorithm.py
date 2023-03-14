"""
Unit tests for the genetic algorithm module.
"""
from pathlib import Path

from swarmrl.rl_protocols.rl_protocol import RLProtocol
from swarmrl.training_routines.genetic_algorithm import GeneticTraining


class TestGeneticAlgorithm:
    """
    Class for testing the genetic algorithm module.
    """

    def test_class_initialization(self):
        """
        Test the creation of the class.
        """
        base_protocol = RLProtocol()
        simulation_runner = None
        genetic_algorithm = GeneticTraining(
            base_protocol,
            simulation_runner,
            number_of_generations=10,
            population_size=10,
            number_of_parents=2,
            number_of_simulations=10,
            output_directory=".",
            routine_name="genetic_algorithm",
        )

        assert genetic_algorithm.base_protocol == base_protocol
        assert genetic_algorithm.number_of_generations == 10
        assert genetic_algorithm.population_size == 10
        assert genetic_algorithm.number_of_parents == 2
        assert genetic_algorithm.number_of_simulations == 10
        assert genetic_algorithm.output_directory == Path("./genetic_algorithm")
