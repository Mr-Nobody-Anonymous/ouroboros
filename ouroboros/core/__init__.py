"""
Core weight manipulation, adapter fine-tuning, architecture mutation, and self-evaluation modules.
"""
from .weight_editor import WeightEditor
from .adapter_engine import AdapterEngine
from .meta_feedback import MetaFeedbackGenerator
from .evaluator import SelfEvaluator
from .task_arithmetic import TaskArithmeticEngine
from .architecture_mutator import ArchitectureMutator
from .self_rewarding import SelfRewardingEngine

__all__ = [
    "WeightEditor",
    "AdapterEngine",
    "MetaFeedbackGenerator",
    "SelfEvaluator",
    "TaskArithmeticEngine",
    "ArchitectureMutator",
    "SelfRewardingEngine"
]
