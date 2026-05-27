"""Lightweight schema validators for NicheLens-ST MVP inputs and outputs.

Mirrors the contract in ``docs/MVP_DESIGN.md``.
"""

from __future__ import annotations

import numpy as np


class SchemaError(ValueError):
    """Raised when an input or output object violates the MVP schema."""


VALID_PROTO_KIND = {"conserved", "sample_specific"}


def validate_inputs(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Validate MVP inputs.

    Raises ``SchemaError`` on any violation. Returns ``None`` on success.
    """
    if X.ndim != 2:
        raise SchemaError(f"X must be 2D; got ndim={X.ndim}")
    if X.dtype != np.float32:
        raise SchemaError(f"X must be float32; got {X.dtype}")
    n_cells = X.shape[0]

    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise SchemaError(
            f"coords must be (n_cells, 2) or (n_cells, 3); got shape={coords.shape}"
        )
    if coords.dtype != np.float32:
        raise SchemaError(f"coords must be float32; got {coords.dtype}")
    if coords.shape[0] != n_cells:
        raise SchemaError(
            f"coords n_cells={coords.shape[0]} != X n_cells={n_cells}"
        )

    if section_id.ndim != 1 or section_id.shape[0] != n_cells:
        raise SchemaError(
            f"section_id must be (n_cells,); got shape={section_id.shape}"
        )
    if section_id.dtype.kind not in ("i", "u"):
        raise SchemaError(f"section_id must be integer dtype; got {section_id.dtype}")

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise SchemaError(f"edges must be (2, n_edges); got shape={edges.shape}")
    if edges.dtype != np.int64:
        raise SchemaError(f"edges must be int64; got {edges.dtype}")
    if edges.size and (edges.min() < 0 or edges.max() >= n_cells):
        raise SchemaError("edges contain indices outside [0, n_cells)")


def validate_outputs(
    H: np.ndarray,
    prototype_id: np.ndarray,
    proto_kind: list[str],
    n_cells: int,
) -> None:
    """Validate MVP outputs.

    ``n_cells`` is passed explicitly so callers can validate outputs against
    a known input size without re-providing the inputs.
    """
    if H.ndim != 2 or H.shape[0] != n_cells:
        raise SchemaError(f"H must be (n_cells, d); got shape={H.shape}")
    if H.dtype != np.float32:
        raise SchemaError(f"H must be float32; got {H.dtype}")
    if H.shape[1] < 1:
        raise SchemaError(f"H embedding dim must be >= 1; got {H.shape[1]}")

    if prototype_id.ndim != 1 or prototype_id.shape[0] != n_cells:
        raise SchemaError(
            f"prototype_id must be (n_cells,); got shape={prototype_id.shape}"
        )
    if prototype_id.dtype != np.int64:
        raise SchemaError(
            f"prototype_id must be int64; got {prototype_id.dtype}"
        )
    if prototype_id.size and prototype_id.min() < 0:
        raise SchemaError("prototype_id must be non-negative")

    n_protos = int(prototype_id.max()) + 1 if prototype_id.size else 0
    if len(proto_kind) != n_protos:
        raise SchemaError(
            f"proto_kind length {len(proto_kind)} != observed n_protos {n_protos}"
        )
    bad = [k for k in proto_kind if k not in VALID_PROTO_KIND]
    if bad:
        raise SchemaError(f"proto_kind contains invalid values: {bad}")
