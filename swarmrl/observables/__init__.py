"""
Module for the different possible observables.
"""
from swarmrl.observables.observable import Observable
from swarmrl.observables.position import PositionObservable
from swarmrl.observables.position_angle import PositionAngleObservable

__all__ = [
    PositionObservable.__name__,
    PositionAngleObservable.__name__,
    Observable.__name__,
]
