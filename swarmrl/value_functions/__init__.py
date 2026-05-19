"""
Package for value functions
"""

from swarmrl.value_functions.expected_returns import ExpectedReturns
from swarmrl.value_functions.generalized_advantage_estimate import GAE
from swarmrl.value_functions.td_return_sac import TDReturnsSAC

__all__ = [ExpectedReturns.__name__, GAE.__name__, TDReturnsSAC.__name__]
