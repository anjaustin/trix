"""
TriX Quantization-Aware Training

Proper infrastructure for training with ternary weights using
Straight-Through Estimator and progressive quantization.
"""

from .quantizers import (
    TernaryQuantizer,
    SoftTernaryQuantizer,
    Top1Gate,
    TriXLinearQAT,
    progressive_quantization_schedule,
    QATTrainer,
)

__all__ = [
    "TernaryQuantizer",
    "SoftTernaryQuantizer",
    "Top1Gate",
    "TriXLinearQAT",
    "progressive_quantization_schedule",
    "QATTrainer",
]
