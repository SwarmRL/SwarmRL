"""
Package for value functions
"""
from swarmrl.value_functions.expected_returns import ExpectedReturns
from swarmrl.value_functions.generalized_advantage_estimate import GAE

__all__ = [ExpectedReturns.__name__,
           GAE.__name__]
