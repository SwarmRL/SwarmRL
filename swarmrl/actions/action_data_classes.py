"""
Data classes for the currently available actions.
"""
from dataclasses import dataclass
from typing import Callable
from swarmrl.actions.actions import TranslateColloid, RotateColloid, Nothing
import numpy as np


@dataclass
class Action:
    """
    A dataclass to describe an action on a colloid.

    Attributes
    ----------
    property : str
            The property that this action updates, e.g. "force" or "torque
    action : Callable
            A callable that updates this property
    """

    property: str
    action: Callable


@dataclass
class Translate(Action):
    """
    Translate a particle
    """

    property = "force"
    action = TranslateColloid(act_force=1.0)


@dataclass
class RotateClockwise(Action):
    """
    Rotate a particle clockwise
    """

    property = "new_direction"
    action = RotateColloid(angle=np.pi / 3)


class RotateCounterClockwise(Action):
    """
    Rotate a particle counter-clockwise.
    """

    property = "new_direction"
    action = RotateColloid(angle=np.pi / 3, clockwise=False)


class DoNothing(Action):
    """
    Do nothing.
    """
    property = "Nothing"
    action = Nothing()
