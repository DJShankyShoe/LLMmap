import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# 1.  Feature extraction
# ---------------------------------------------------------------------
@torch.inference_mode()
def infer_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the *feature extractor* on every sample in `loader`.

    Returns
    -------
    labels : np.ndarray  shape (N,)     – class index per sample
    feats  : np.ndarray  shape (N, F)   – extracted embedding
    """
    model.eval().to(device)

    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)                  # (B, F)
        feats.append(out.cpu())
        labels.append(y.cpu())

    feats  = torch.cat(feats).numpy()   # (N, F)
    labels = torch.cat(labels).numpy()  # (N,)
    return labels, feats


# ---------------------------------------------------------------------
# 2.  Build per-class templates  (simple mean)
# ---------------------------------------------------------------------
def build_templates(
    labels: np.ndarray,
    feats: np.ndarray,
) -> np.ndarray:
    """
    Compute a mean feature vector for every class ID that appears.

    Returns
    -------
    templates : np.ndarray  shape (C, F)  – C = max(label)+1
    """
    num_classes = int(labels.max()) + 1
    F           = feats.shape[1]
    templates   = np.zeros((num_classes, F), dtype=feats.dtype)

    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            templates[c] = feats[mask].mean(axis=0)
        else:                       # class `c` absent in training split
            templates[c] = np.nan   # optional: leave NaNs as sentinel
    return templates


# ---------------------------------------------------------------------
# 3.  Classification by nearest template
# ---------------------------------------------------------------------
def predict_by_templates(
    feats: np.ndarray,
    templates: np.ndarray,
) -> np.ndarray:
    """
    Assign each feature to the class whose template is *closest* (L2).

    Returns
    -------
    pred_labels : np.ndarray  shape (N,)
    """
    # (N, 1, F) - (1, C, F) → (N, C)
    dists = np.linalg.norm(feats[:, None, :] - templates[None, :, :], axis=-1)
    return dists.argmin(axis=1)


# ---------------------------------------------------------------------
# 4.  Wrapper that does everything and reports accuracy
# ---------------------------------------------------------------------
def template_generation(
    feature_extractor: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    l2_normalize: bool = False,
    threshold_percentile: float = 95.0,
) -> Dict[str, object]:
    """
    Returns
    -------
    dict with keys:
        y_true, y_pred      – ground-truth / predicted labels (test set)
        templates           – class centroids (from the training split)
        accuracy            – top-1 accuracy in [0,1]
        top3_accuracy       – top-3 accuracy
        open_set_threshold  – `threshold_percentile`-th percentile of the distance between a
                              test trace and its *own* template; a live target farther than this
                              from every template is probably an unknown model.
    """
    def _norm(f):
        return f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-12) if l2_normalize else f

    y_train, f_train = infer_features(feature_extractor, train_loader, device)
    templates = build_templates(y_train, _norm(f_train))

    y_test, f_test = infer_features(feature_extractor, test_loader, device)
    f_test = _norm(f_test)
    dists = np.linalg.norm(f_test[:, None, :] - templates[None, :, :], axis=-1)
    dists = np.where(np.isnan(dists), np.inf, dists)
    order = dists.argsort(axis=1)
    y_pred = order[:, 0]

    acc = float((y_pred == y_test).mean())
    top3 = float((order[:, :3] == y_test[:, None]).any(1).mean())
    own = dists[np.arange(len(y_test)), y_test]
    ths = float(np.percentile(own[np.isfinite(own)], threshold_percentile)) if len(own) else None

    per_class = {}
    for c in np.unique(y_test):
        m = y_test == c
        per_class[int(c)] = float((y_pred[m] == c).mean())

    return dict(
        y_true=y_test,
        y_pred=y_pred,
        templates=templates,
        accuracy=acc,
        top3_accuracy=top3,
        open_set_threshold=ths,
        per_class_accuracy=per_class,
    )
