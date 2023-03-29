"""
Module for the different possible observables.
"""
from swarmrl.observables.col_graph import ColGraph
from swarmrl.observables.concentration_field import ConcentrationField
from swarmrl.observables.director import Director
from swarmrl.observables.multi_sensing import MultiSensing
from swarmrl.observables.observable import Observable
from swarmrl.observables.particle_sensing import ParticleSensing
from swarmrl.observables.position import PositionObservable
from swarmrl.observables.subdivided_vision_cones import SubdividedVisionCones

__all__ = [
    PositionObservable.__name__,
    Director.__name__,
    MultiSensing.__name__,
    Observable.__name__,
    ConcentrationField.__name__,
    ParticleSensing.__name__,
    SubdividedVisionCones.__name__,
    ColGraph.__name__,
]
