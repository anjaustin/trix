"""Developer-first drop-in wrapper.

This is the narrow-waist API for Mesa 14.

Defaults:
- forward(x) returns only the output tensor (drop-in for standard FFNs)
- optional return_aux returns (out, routing_info, aux_losses)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .compiled_dispatch import CompiledDispatch
from .lifecycle import RoutingLifecycleV1
from .sparse_lookup_v2 import SparseLookupFFNv2
from .bundle import export_address_bundle


@dataclass
class DropInConfig:
    d_model: int
    num_tiles: int = 8
    tiles_per_cluster: int = 4
    grid_size: int = 16
    dropout: float = 0.0
    use_score_calibration: bool = False
    routing_backend: str = "hierarchical_dot"  # or flat_popcount


class DropInFFN(nn.Module):
    """A developer-first FFN replacement.

    Modes:
    - dynamic: uses ffn.forward routing
    - contracted: uses compiled dispatch when a class_hint is provided
    - packed: uses flat_popcount routing backend inside the module
    """

    def __init__(
        self,
        cfg: DropInConfig,
        *,
        mode: str = "dynamic",  # dynamic|contracted|packed
    ):
        super().__init__()
        if mode not in {"dynamic", "contracted", "packed"}:
            raise ValueError("mode must be one of: dynamic, contracted, packed")

        routing_backend = cfg.routing_backend
        if mode == "packed":
            routing_backend = "flat_popcount"

        self.cfg = cfg
        self.mode = mode

        self.ffn = SparseLookupFFNv2(
            d_model=cfg.d_model,
            num_tiles=cfg.num_tiles,
            tiles_per_cluster=cfg.tiles_per_cluster,
            grid_size=cfg.grid_size,
            compress_hidden=None,
            dropout=cfg.dropout,
            use_score_calibration=cfg.use_score_calibration,
            routing_backend=routing_backend,
        )

        self.compiled: Optional[CompiledDispatch] = None
        if mode == "contracted":
            self.compiled = CompiledDispatch(self.ffn)

    def forward(
        self,
        x: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        class_hint: Optional[int] = None,
        confidence: float = 1.0,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        if (
            self.mode == "contracted"
            and self.compiled is not None
            and class_hint is not None
        ):
            out, info, aux = self.compiled.forward(
                x,
                labels=labels,
                class_hint=int(class_hint),
                confidence=float(confidence),
            )
        else:
            out, info, aux = self.ffn.forward(x, labels=labels)

        # Normalize a couple of fields so downstream telemetry is consistent.
        info.setdefault("compiled", False)
        info.setdefault("guard_failed", False)

        return (out, info, aux) if return_aux else out

    def compile_stable(
        self,
        *,
        threshold: float = 0.3,
        min_confidence: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        if self.compiled is None:
            self.compiled = CompiledDispatch(self.ffn)
        return self.compiled.compile_stable(
            threshold=float(threshold),
            min_confidence=float(min_confidence),
            num_classes=num_classes,
        )

    def lifecycle(self) -> RoutingLifecycleV1:
        return RoutingLifecycleV1(self.ffn, compiled=self.compiled)

    def export_bundle(
        self,
        *,
        outdir: str,
        include_state_dict: bool = True,
        include_validation: bool = True,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        validation = None
        if include_validation and self.compiled is not None:
            from .bundle import validate_compiled_semantics

            validation = validate_compiled_semantics(
                ffn=self.ffn, compiled=self.compiled, samples=64, seed=0
            )

        return export_address_bundle(
            ffn=self.ffn,
            compiled=self.compiled,
            outdir=outdir,
            include_state_dict=include_state_dict,
            validation=validation,
            extra_meta=extra_meta,
        )
