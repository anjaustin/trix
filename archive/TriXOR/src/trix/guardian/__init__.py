# HALO - Homeo-Adaptive Learning Observer
# Mesa 12: Guardian Angel Architecture
#
# "Who needs Human Reinforcement Learning Feedback 
#  when you have a Homeo-Adaptive Learning Observer?!"
#
# "All things are connected through gentleness."
# "Wrong is just a signal. Distributed entropy signaling the correct direction."
# "It is the ultimate form of Love."
#
# RLHF is dead. Long live HALO.

from .programmable_tile import ProgrammableTile, ProgrammableTileBank
from .observer import ObservationFrame, StateEncoder, ObserverModel
from .reflector import SuperpositionedReflector, XORReflector
from .guardian import GuardianAngel
from .training import GuardedTrainer
from .pipeline import HALOPipeline, Phase, EntropicHarmonyLoss, JourneyContext

__all__ = [
    # Core
    'ProgrammableTile',
    'ProgrammableTileBank', 
    'ObservationFrame',
    'StateEncoder',
    'ObserverModel',
    'SuperpositionedReflector',
    'XORReflector',
    'GuardianAngel',
    'GuardedTrainer',
    # Pipeline
    'HALOPipeline',
    'Phase',
    'EntropicHarmonyLoss',
    'JourneyContext',
]
