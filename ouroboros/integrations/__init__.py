"""
External model editing, weight merging, and fine-tuning integrations.
"""
from .easyedit_bridge import EasyEditBridge
from .mergekit_bridge import MergeKitBridge

__all__ = ["EasyEditBridge", "MergeKitBridge"]
