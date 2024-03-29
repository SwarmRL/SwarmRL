"""
__init__ for the tasks module.
"""

from swarmrl.tasks import object_movement, searching
from swarmrl.tasks.multi_tasking import MultiTasking
from swarmrl.tasks.task import Task

__all__ = [
    searching.__name__,
    object_movement.__name__,
    Task.__name__,
    MultiTasking.__name__,
]
