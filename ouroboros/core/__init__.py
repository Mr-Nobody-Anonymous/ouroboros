"""
Core weight manipulation, adapter fine-tuning, architecture mutation, self-evolution, memory, and test-time compute modules.
"""
from .weight_editor import WeightEditor
from .adapter_engine import AdapterEngine
from .meta_feedback import MetaFeedbackGenerator
from .evaluator import SelfEvaluator
from .task_arithmetic import TaskArithmeticEngine
from .architecture_mutator import ArchitectureMutator
from .self_rewarding import SelfRewardingEngine
from .anchor_memory import AnchorMemoryBank
from .adversarial_critic import AdversarialCritic
from .code_mutator import CodeMutator
from .pareto_pruner import ParetoCheckpointPruner
from .weight_space_programmer import WeightSpaceProgrammer
from .test_time_compute import TestTimeComputeEngine

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
    "TestTimeComputeEngine"
]
