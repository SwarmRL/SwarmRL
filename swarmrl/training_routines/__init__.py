"""
Module for special training routines.
"""
from swarmrl.training_routines.ensemble_submit import EnsembleTraining
from swarmrl.training_routines.genetic_algorithm import GeneticTraining

__all__ = [GeneticTraining.__name__, EnsembleTraining.__name__]
