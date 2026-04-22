# Copyright (c) 2026 BAAI. All rights reserved.

"""
GCU backend for vllm-plugin-FL dispatch (Enflame / ``torch.gcu``).
"""

from .gcu import GCUBackend

__all__ = ["GCUBackend"]
