"""
Core weight manipulation, adapter fine-tuning, architecture mutation, self-evolution, memory, and test-time compute modules.
"""

from .adapter_engine import AdapterEngine
from .adversarial_critic import AdversarialCritic
from .anchor_memory import AnchorMemoryBank
from .architecture_mutator import ArchitectureMutator
from .code_mutator import CodeMutator
from .evaluator import SelfEvaluator
from .meta_feedback import MetaFeedbackGenerator
from .pareto_pruner import ParetoCheckpointPruner
from .self_rewarding import SelfRewardingEngine
from .task_arithmetic import TaskArithmeticEngine
from .test_time_compute import TestTimeComputeEngine
from .weight_editor import WeightEditor
from .weight_space_programmer import WeightSpaceProgrammer

__all__ = [
    "WeightEditor",
    "AdapterEngine",
    "MetaFeedbackGenerator",
    "SelfEvaluator",
    "TaskArithmeticEngine",
    "ArchitectureMutator",
    "SelfRewardingEngine",
    "AnchorMemoryBank",
    "AdversarialCritic",
    "CodeMutator",
    "ParetoCheckpointPruner",
    "WeightSpaceProgrammer",
    "TestTimeComputeEngine",
]
