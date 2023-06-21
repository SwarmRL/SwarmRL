"""
Init file for object movement tasks.
"""
from swarmrl.tasks.object_movement.drug_delivery import DrugDealing
from swarmrl.tasks.object_movement.rod_rotation import RotateRod

__all__ = [RotateRod.__name__, DrugDealing.__name__]
