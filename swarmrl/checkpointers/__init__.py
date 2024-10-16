"""
Package to hold all SwarmRL Checkpointers.
"""

from swarmrl.checkpointers.backup_checkpointer import BackupCheckpointer
from swarmrl.checkpointers.best_checkpointer import BestRewardCheckpointer
from swarmrl.checkpointers.checkpointer import Checkpointer
from swarmrl.checkpointers.goal_checkpointer import GoalCheckpointer
from swarmrl.checkpointers.regular_checkpointer import RegularCheckpointer

__all__ = [
    BackupCheckpointer.__name__,
    BestRewardCheckpointer.__name__,
    GoalCheckpointer.__name__,
    RegularCheckpointer.__name__,
    Checkpointer.__name__,
]
