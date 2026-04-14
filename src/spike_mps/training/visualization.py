from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
from tensor_network_viz import show_tensor_network

from spike_mps.training.checkpoints import load_model_from_checkpoint


def visualize_checkpoint(
    *,
    checkpoint_path: Path,
    show: bool = True,
) -> tuple[object, object]:
    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" in backend:
            raise RuntimeError(
                "Interactive visualization requires a GUI backend. "
                "Run in a GUI environment or pass --no-show."
            )

    model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Using a non-tuple sequence for multidimensional indexing.*",
            category=UserWarning,
        )
        fig, ax = show_tensor_network(
            model.network,
            engine="tensorkrowch",
            show=show,
        )
    return fig, ax
