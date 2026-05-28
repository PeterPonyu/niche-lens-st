"""NicheLens-ST top-level package.

The core model (contrastive niche-token encoder + prototype/separation head)
lives in :mod:`nichelens_st.model` and :mod:`nichelens_st.encoder`. The encoder
uses PyTorch, declared as the optional ``[model]`` extra
(``pip install nichelens-st[model]``); the base package stays dependency-light
(numpy/scipy) and importing it never requires torch. Biology and performance
claims remain planned (see CLAIM_LEDGER.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "NicheModelConfig",
    "NicheModelResult",
    "fit_niche_model",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import NicheModelConfig, NicheModelResult, fit_niche_model


def __getattr__(name: str):
    """Lazily expose the model API without eagerly importing it.

    ``nichelens_st.model`` itself gates torch, so this stays import-safe even
    without the optional extra installed.
    """
    if name in {"NicheModelConfig", "NicheModelResult", "fit_niche_model"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
