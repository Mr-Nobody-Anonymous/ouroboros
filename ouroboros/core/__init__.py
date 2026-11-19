"""
Core weight manipulation, adapter fine-tuning, and self-evaluation modules.
"""
from .weight_editor import WeightEditor
from .adapter_engine import AdapterEngine
from .meta_feedback import MetaFeedbackGenerator
from .evaluator import SelfEvaluator

__all__ = ["WeightEditor", "AdapterEngine", "MetaFeedbackGenerator", "SelfEvaluator"]
