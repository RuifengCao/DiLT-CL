"""Gromov-Wasserstein representation-similarity analysis.

These utilities are *optional* - ``compute_gw_distance`` depends on the POT
library (``pip install POT``) and scikit-learn. Both are imported lazily so
this module always imports cleanly; if a dependency is missing the function
degrades gracefully by returning ``None``.
"""

import numpy as np
import torch
import torch.nn as nn


def extract_layer_representations(model, dataloader, layer_name=None,
                                  device="cuda", max_samples=1000):
    """Extract intermediate-layer representations for a set of inputs.

    If ``layer_name`` is ``None`` the penultimate linear layer (``fc1`` when
    present, otherwise the last ``nn.Linear``) is used.

    Args:
        model: PyTorch model.
        dataloader: DataLoader to pull inputs from.
        layer_name: Name of the module to hook (optional).
        device: Device to run on.
        max_samples: Maximum number of samples to process.

    Returns:
        A numpy array of shape ``(n_samples, feature_dim)``.
    """
    model.eval()

    # Resolve the target module to hook.
    if layer_name is None:
        target_module = None
        for name, module in model.named_modules():
            if name == "fc1":
                target_module = module
                break
        if target_module is None:
            for _, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    target_module = module
    else:
        target_module = dict(model.named_modules())[layer_name]

    if target_module is None:
        raise ValueError("Could not resolve a layer to extract from.")

    activation = {}

    def hook(_module, _inp, output):
        activation["target"] = output.detach()

    handle = target_module.register_forward_hook(hook)

    representations = []
    total_samples = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _ = model(data)

            feat = activation["target"]
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)

            representations.append(feat.cpu().numpy())
            total_samples += data.size(0)
            if total_samples >= max_samples:
                break

    handle.remove()
    return np.concatenate(representations, axis=0)[:max_samples]


def compute_gw_distance(repr1, repr2, reg=0.01, max_samples=200):
    """Entropic Gromov-Wasserstein distance between two representation sets.

    Robust to POT version differences and missing optional dependencies.

    Args:
        repr1: Array of shape ``(n1, d1)``.
        repr2: Array of shape ``(n2, d2)``.
        reg: Entropic regularisation strength.
        max_samples: Cap on samples per side (for tractability).

    Returns:
        The GW distance as a float, or ``None`` if it cannot be computed.
    """
    try:
        import ot
        from sklearn.metrics import pairwise_distances
    except ImportError:
        print("[Warning] POT and/or scikit-learn not found. "
              "Install with: pip install POT scikit-learn")
        return None

    n = min(len(repr1), len(repr2), max_samples)
    if n < 2:
        return None
    if len(repr1) > n:
        repr1 = repr1[np.random.choice(len(repr1), n, replace=False)]
    if len(repr2) > n:
        repr2 = repr2[np.random.choice(len(repr2), n, replace=False)]

    # Intra-set distance matrices, normalised to [0, 1].
    C1 = pairwise_distances(repr1, metric="euclidean")
    C2 = pairwise_distances(repr2, metric="euclidean")
    C1 = C1 / (C1.max() + 1e-8)
    C2 = C2 / (C2.max() + 1e-8)

    p = ot.unif(len(repr1))
    q = ot.unif(len(repr2))

    try:
        result = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q, "square_loss", epsilon=reg, log=True
        )
    except Exception as exc:  # noqa: BLE001 - tolerate POT API differences
        print(f"[Warning] GW computation failed: {exc}")
        return None

    # ``log=True`` returns ``(coupling, log_dict)`` in recent POT releases.
    if isinstance(result, tuple) and len(result) == 2:
        _, log = result
        if isinstance(log, dict) and "gw_dist" in log:
            return float(log["gw_dist"])

    # Fallback for versions that return the distance directly.
    try:
        return float(result)
    except (TypeError, ValueError):
        return None
