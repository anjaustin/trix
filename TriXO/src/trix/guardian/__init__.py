# Guardian Angel Architecture
# Mesa 12: Observer + Programmable Tiles + Superpositioned Reflector
#
# "All things are connected through gentleness."
# "Wrong is just a signal. Distributed entropy signaling the correct direction."
# "It is the ultimate form of Love."

from .programmable_tile import ProgrammableTile, ProgrammableTileBank
from .observer import ObservationFrame, StateEncoder, ObserverModel
from .reflector import SuperpositionedReflector, XORReflector
from .guardian import GuardianAngel
from .training import GuardedTrainer

__all__ = [
    'ProgrammableTile',
    'ProgrammableTileBank', 
    'ObservationFrame',
    'StateEncoder',
    'ObserverModel',
    'SuperpositionedReflector',
    'XORReflector',
    'GuardianAngel',
    'GuardedTrainer',
]
