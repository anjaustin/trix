"""Routing lifecycle v1.

This module provides a thin, explicit lifecycle wrapper around routing-driven
FFNs (currently: SparseLookupFFNv2) and optional CompiledDispatch.

Goal: make observe/edit/compile/guard/monitor a single, testable surface.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .compiled_dispatch import CompiledDispatch
from .sparse_lookup_v2 import SparseLookupFFNv2


@dataclass
class LifecycleGuardConfig:
    # If margin is small (ambiguous routing), flag as near-tie.
    # Note: margin units depend on the score domain.
    near_tie_margin: float = 1e-3


def _jsonl_append(path: os.PathLike[str] | str, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _route_margin_stats(scores: torch.Tensor) -> Dict[str, float]:
    """Compute tie_rate and margin_mean from a [N, K] score tensor."""
    if scores.numel() == 0:
        return {"tie_rate": 0.0, "near_tie_rate": 0.0, "margin_mean": 0.0}

    top2 = torch.topk(scores, k=2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).to(torch.float32)

    # Exact ties are rare for float scores; keep for completeness.
    tie = margin == 0
    tie_rate = tie.to(torch.float32).mean().item()
    margin_mean = margin.mean().item()

    return {"tie_rate": tie_rate, "margin_mean": margin_mean}


def _resample_ternary_like(
    signatures: torch.Tensor, flip_prob: float, seed: int
) -> torch.Tensor:
    """Resample coordinates to {-1,0,+1} with probability flip_prob."""
    if flip_prob <= 0:
        return signatures

    flip_prob = float(max(0.0, min(1.0, flip_prob)))
    g = torch.Generator(device=signatures.device)
    g.manual_seed(int(seed))

    # Bernoulli mask
    mask = torch.rand_like(signatures, dtype=torch.float32, generator=g) < flip_prob
    # Uniform choice among {-1,0,+1}
    r = torch.randint(0, 3, signatures.shape, device=signatures.device, generator=g)
    resampled = torch.where(
        r == 0,
        torch.tensor(-1.0, device=signatures.device),
        torch.where(
            r == 1,
            torch.tensor(0.0, device=signatures.device),
            torch.tensor(1.0, device=signatures.device),
        ),
    )
    return torch.where(mask, resampled.to(signatures.dtype), signatures)


class RoutingLifecycleV1:
    """Lifecycle wrapper for SparseLookupFFNv2 (+ optional CompiledDispatch)."""

    def __init__(
        self,
        ffn: SparseLookupFFNv2,
        *,
        compiled: Optional[CompiledDispatch] = None,
        guard: Optional[LifecycleGuardConfig] = None,
    ):
        self.ffn = ffn
        self.compiled = compiled
        self.guard = guard or LifecycleGuardConfig()

    # ---------------------------------------------------------------------
    # Observe
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def observe(
        self,
        x: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        class_hint: Optional[int] = None,
        confidence: float = 1.0,
        jsonl_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run forward pass and emit a JSONL telemetry record (optional)."""

        t0 = time.perf_counter()
        if self.compiled is not None:
            out, routing_info, aux = self.compiled.forward(
                x, class_hint=class_hint, confidence=float(confidence), labels=labels
            )
        else:
            out, routing_info, aux = self.ffn.forward(x, labels=labels)
            routing_info["compiled"] = False
            routing_info["guard_failed"] = False
        t1 = time.perf_counter()

        # Compute routing stats in a way that matches the module's hierarchical routing.
        tie_rate = 0.0
        near_tie_rate = 0.0
        margin_mean = 0.0
        try:
            B, T, D = x.shape
            x_flat = self.ffn.norm(x).view(-1, D)
            sigs = self.ffn.signatures
            tile_idx, all_scores = self.ffn.route(x_flat, sigs, return_scores=True)

            # all_scores is sparse-filled per selected cluster; for a stable signal,
            # compute margin over the scores that were actually filled.
            # We approximate this by restricting to tiles that have any non-zero score.
            if all_scores is not None:
                nonzero_cols = all_scores.abs().sum(dim=0) > 0
                if nonzero_cols.any():
                    stats = _route_margin_stats(all_scores[:, nonzero_cols])
                    tie_rate = float(stats["tie_rate"])
                    margin_mean = float(stats["margin_mean"])

                    margin = torch.topk(all_scores[:, nonzero_cols], k=2, dim=-1).values
                    m = (margin[:, 0] - margin[:, 1]).to(torch.float32)
                    near_tie_rate = (
                        (m <= float(self.guard.near_tie_margin)).float().mean().item()
                    )

            # Ensure routing_info has tile_idx even for compiled fallback.
            if "tile_idx" not in routing_info:
                routing_info["tile_idx"] = tile_idx.view(B, T)
        except Exception:
            # Telemetry should not break inference.
            pass

        fallback_applied = bool(routing_info.get("guard_failed", False))
        fallback_reason = "compiled_guard_failed" if fallback_applied else None

        record = {
            "schema_version": 1,
            "event": "routing",
            "run_id": run_id or "python-observe",
            "seed": None,
            "address_type": "tile_id",
            "tiles": int(getattr(self.ffn, "num_tiles", 0)),
            "dim": int(getattr(self.ffn, "d_model", 0)),
            "inputs": int(x.numel() // max(1, int(getattr(self.ffn, "d_model", 1)))),
            "routing_ms": (t1 - t0) * 1000.0,
            "routes_per_s": (x.shape[0] * x.shape[1]) / max(1e-9, (t1 - t0)),
            "tie_rate": tie_rate,
            "near_tie_rate": near_tie_rate,
            "margin_mean": margin_mean,
            "tie_break": None,
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
            "compiled": bool(routing_info.get("compiled", False)),
            "routing_backend": routing_info.get("routing_backend"),
        }

        try:
            stats_fn = getattr(self.ffn, "get_signature_compression_stats", None)
            if callable(stats_fn):
                record["signature_compression"] = stats_fn()
        except Exception:
            record["signature_compression"] = None

        if jsonl_path is not None:
            _jsonl_append(jsonl_path, record)

        return out, routing_info, aux, record

    # ---------------------------------------------------------------------
    # Edit (surgery)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def edit_insert_signature(
        self,
        *,
        tile_idx: int,
        signature: torch.Tensor,
        freeze: bool = True,
        tag: str = "",
    ) -> Dict[str, Any]:
        """Insert signature with pre/post snapshot for reversibility."""
        if signature.shape != (self.ffn.d_model,):
            raise ValueError(f"signature must have shape ({self.ffn.d_model},)")

        prev_sig = self.ffn.signatures_raw[tile_idx].detach().clone()
        prev_frozen = self.ffn.is_frozen(tile_idx)

        # Basic validation: keep values in [-1, 1] to avoid surprising scale.
        if signature.min().item() < -1.0 or signature.max().item() > 1.0:
            raise ValueError("signature values must be within [-1, 1] for v1 surgery")

        self.ffn.insert_signature(
            tile_idx=tile_idx, signature=signature, freeze=freeze, tag=tag
        )

        post = self.ffn.get_signature_analysis(tile_idx)
        op = {
            "op": "insert_signature",
            "tile_idx": int(tile_idx),
            "tag": tag,
            "prev": {"signature_raw": prev_sig, "frozen": prev_frozen},
            "post": post,
        }
        return op

    @torch.no_grad()
    def undo(self, op: Dict[str, Any]) -> None:
        """Undo an edit operation record produced by this lifecycle."""
        if op.get("op") != "insert_signature":
            raise ValueError("unsupported op for undo")

        tile_idx = int(op["tile_idx"])
        prev_sig = op["prev"]["signature_raw"]
        prev_frozen = bool(op["prev"]["frozen"])

        self.ffn.insert_signature(
            tile_idx=tile_idx, signature=prev_sig, freeze=False, tag="undo"
        )
        if prev_frozen:
            self.ffn.freeze_signature(tile_idx)
        else:
            self.ffn.unfreeze_signature(tile_idx)

    # ---------------------------------------------------------------------
    # Compile
    # ---------------------------------------------------------------------

    def ensure_compiled(self) -> CompiledDispatch:
        if self.compiled is None:
            self.compiled = CompiledDispatch(self.ffn)
        return self.compiled

    def compile_stable(
        self,
        *,
        threshold: float = 0.5,
        min_confidence: float = 0.5,
        num_classes: Optional[int] = None,
    ) -> Dict[int, Any]:
        comp = self.ensure_compiled()
        return comp.compile_stable(
            threshold=threshold, min_confidence=min_confidence, num_classes=num_classes
        )

    def export_dispatch_table(self) -> Dict[str, Any]:
        comp = self.ensure_compiled()
        return comp.export_dispatch_table()

    def import_dispatch_table(self, data: Dict[str, Any]) -> None:
        comp = self.ensure_compiled()
        comp.import_dispatch_table(data)

    # ---------------------------------------------------------------------
    # Guard / Monitor
    # ---------------------------------------------------------------------

    def monitor(self) -> Dict[str, Any]:
        """Return a consolidated view of current routing + compiled stats."""
        out: Dict[str, Any] = {
            "routing": self.ffn.get_routing_stats(),
            "islands": self.ffn.get_island_stats(),
        }
        if self.compiled is not None:
            out["compiled"] = self.compiled.get_stats()
        return out

    @torch.no_grad()
    def stability_probe(
        self,
        x: torch.Tensor,
        *,
        flip_prob: float = 0.001,
        seed: int = 1,
        jsonl_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Measure churn under controlled signature perturbation."""
        B, T, D = x.shape
        x_flat = self.ffn.norm(x).view(-1, D)
        sig0 = self.ffn.signatures
        r0, _ = self.ffn.route(x_flat, sig0, return_scores=False)

        sig1 = _resample_ternary_like(sig0, flip_prob=flip_prob, seed=seed)
        r1, _ = self.ffn.route(x_flat, sig1, return_scores=False)

        churn = (r0 != r1).float().mean().item() if r0.numel() else 0.0

        record = {
            "schema_version": 1,
            "event": "stability",
            "run_id": run_id or "python-stability",
            "seed": int(seed),
            "address_type": "tile_id",
            "tiles": int(getattr(self.ffn, "num_tiles", 0)),
            "dim": int(getattr(self.ffn, "d_model", 0)),
            "inputs": int(B * T),
            "flip_prob": float(flip_prob),
            "churn": float(churn),
        }
        if jsonl_path is not None:
            _jsonl_append(jsonl_path, record)
        return record
