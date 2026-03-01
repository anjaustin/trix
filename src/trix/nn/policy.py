"""Mesa 15: runtime address policy enforcement.

This is intentionally minimal: integrity (tamper detection) is separate from
identity/signing. Policies here are local allow/deny rules applied at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set


@dataclass(frozen=True)
class AddressPolicyV1:
    """Allow/deny policy for tile_id addresses.

    Semantics:
    - If allow_tiles is provided, only those tiles are allowed.
    - deny_tiles always overrides allow_tiles.
    - on_violation:
        - "fallback": attempt to reroute to an allowed tile
        - "fail": raise a RuntimeError
        - "noop": treat disallowed routes as no-op (model-dependent)
          (currently implemented as fallback when possible)
    """

    allow_tiles: Optional[Set[int]] = None
    deny_tiles: Set[int] = None  # type: ignore[assignment]
    on_violation: str = "fallback"

    def __post_init__(self) -> None:
        d = set(self.deny_tiles or set())
        object.__setattr__(self, "deny_tiles", d)
        if self.allow_tiles is not None:
            object.__setattr__(self, "allow_tiles", set(self.allow_tiles))
        if self.on_violation not in {"fallback", "fail", "noop"}:
            raise ValueError("on_violation must be one of: fallback, fail, noop")

    @staticmethod
    def from_lists(
        *,
        allow_tiles: Optional[Iterable[int]] = None,
        deny_tiles: Optional[Iterable[int]] = None,
        on_violation: str = "fallback",
    ) -> "AddressPolicyV1":
        return AddressPolicyV1(
            allow_tiles=set(allow_tiles) if allow_tiles is not None else None,
            deny_tiles=set(deny_tiles) if deny_tiles is not None else set(),
            on_violation=on_violation,
        )

    def is_allowed(self, tile_idx: int) -> bool:
        ti = int(tile_idx)
        if ti in self.deny_tiles:
            return False
        if self.allow_tiles is None:
            return True
        return ti in self.allow_tiles

    def has_any_allow(self, *, num_tiles: int) -> bool:
        if self.allow_tiles is not None:
            return any(
                0 <= int(t) < int(num_tiles) and t not in self.deny_tiles
                for t in self.allow_tiles
            )
        # Otherwise: allow anything not explicitly denied.
        return len(self.deny_tiles) < int(num_tiles)
