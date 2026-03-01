"""
Adaptive Training Observer Module

This module provides tools for observing and adapting training dynamics:

- TrainingObserver: Monitors training and applies gentle interventions
- AdaptiveTrainingPipeline: Multi-phase training with entropy-aware exploration
- ProgrammableTile: Tiles with read/write interface for introspection

Experimental: This module contains research code for adaptive training.
The core trix.nn and trix.kernel modules are production-ready; this is not.
"""

from .programmable_tile import ProgrammableTile, ProgrammableTileBank
from .observer import ObservationFrame, StateEncoder, ObserverModel
from .reflector import SuperpositionedReflector, XORReflector
from .guardian import TrainingObserver, GuardianAngel  # GuardianAngel kept as alias
from .training import GuardedTrainer
from .pipeline import AdaptiveTrainingPipeline, HALOPipeline, Phase, EntropyBalanceLoss, EntropicHarmonyLoss, JourneyContext

__all__ = [
    # Core components
    'ProgrammableTile',
    'ProgrammableTileBank',
    'ObservationFrame',
    'StateEncoder',
    'ObserverModel',
    'SuperpositionedReflector',
    'XORReflector',
    # Training observer
    'TrainingObserver',
    'GuardianAngel',  # Alias for backwards compatibility
    'GuardedTrainer',
    # Adaptive pipeline
    'AdaptiveTrainingPipeline',
    'HALOPipeline',  # Alias for backwards compatibility
    'Phase',
    'EntropyBalanceLoss',
    'EntropicHarmonyLoss',  # Alias for backwards compatibility
    'JourneyContext',
]
