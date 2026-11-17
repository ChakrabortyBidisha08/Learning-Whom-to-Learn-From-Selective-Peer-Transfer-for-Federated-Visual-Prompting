from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


_EPS = 1e-12


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _sqrtm_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition."""
    matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.clip(evals, eps, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def bures_gaussian_distance(mu_a, cov_a, mu_b, cov_b, eps: float = 1e-9) -> float:
    """2-Wasserstein distance between Gaussian approximations N(mu_a,cov_a), N(mu_b,cov_b)."""
    mu_a = _as_numpy(mu_a).reshape(-1)
    mu_b = _as_numpy(mu_b).reshape(-1)
    cov_a = _as_numpy(cov_a)
    cov_b = _as_numpy(cov_b)

    cov_a = 0.5 * (cov_a + cov_a.T) + eps * np.eye(cov_a.shape[0])
    cov_b = 0.5 * (cov_b + cov_b.T) + eps * np.eye(cov_b.shape[0])

    sqrt_a = _sqrtm_psd(cov_a, eps=eps)
    middle = sqrt_a @ cov_b @ sqrt_a
    sqrt_middle = _sqrtm_psd(middle, eps=eps)

    mean_term = float(np.dot(mu_a - mu_b, mu_a - mu_b))
    cov_term = float(np.trace(cov_a + cov_b - 2.0 * sqrt_middle))
    # Numerical roundoff can make the trace term microscopically negative.
    return math.sqrt(max(mean_term + cov_term, 0.0))




def _prepare_gaussian(mu, cov, eps: float = 1e-9) -> Dict[str, np.ndarray]:
    """Precompute client-specific terms reused in all Bures pair comparisons."""
    mu = _as_numpy(mu).reshape(-1)
    cov = _as_numpy(cov)
    cov = 0.5 * (cov + cov.T) + eps * np.eye(cov.shape[0])
    return {
        "mu": mu,
        "cov": cov,
        "sqrt_cov": _sqrtm_psd(cov, eps=eps),
        "trace_cov": np.asarray(float(np.trace(cov))),
    }


def _bures_gaussian_distance_prepared(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], eps: float = 1e-9) -> float:
    """Fast Bures/2-Wasserstein distance using one cached covariance square root."""
    delta = a["mu"] - b["mu"]
    middle = a["sqrt_cov"] @ b["cov"] @ a["sqrt_cov"]
    middle = 0.5 * (middle + middle.T)
    eigvals = np.linalg.eigvalsh(middle)
    cross_trace = float(np.sqrt(np.clip(eigvals, eps, None)).sum())
    value = float(np.dot(delta, delta)) + float(a["trace_cov"]) + float(b["trace_cov"]) - 2.0 * cross_trace
    return math.sqrt(max(value, 0.0))


def prompt_update_angle_distance(prompt_delta_a, prompt_delta_b, prompt_a=None, prompt_b=None) -> float:
    """Angular distance between client visual-prompt adaptation directions.

    When both deltas are essentially zero (e.g. before meaningful local adaptation), fall
    back to the raw prompt vectors rather than returning an unstable angle.
    """
    a = _as_numpy(prompt_delta_a).reshape(-1)
    b = _as_numpy(prompt_delta_b).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-10 or nb < 1e-10:
        if prompt_a is None or prompt_b is None:
            return 0.0
        a = _as_numpy(prompt_a).reshape(-1)
        b = _as_numpy(prompt_b).reshape(-1)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

    if na < 1e-10 or nb < 1e-10:
        return 0.0

    cosine = float(np.dot(a, b) / max(na * nb, _EPS))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 1.0 - cosine


def projected_quantile_distance(q_a, q_b) -> float:
    a = _as_numpy(q_a)
    b = _as_numpy(q_b)
    if a.shape != b.shape:
        raise ValueError(f"quantile shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _robust_scale_distance_matrix(D: np.ndarray) -> np.ndarray:
    """Scale a distance view by its median positive pairwise distance."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    upper = D[np.triu_indices(n, k=1)]
    positive = upper[np.isfinite(upper) & (upper > 1e-12)]
    scale = float(np.median(positive)) if positive.size else 1.0
    return D / max(scale, 1e-12)


def _pairwise_view(n: int, distance_fn) -> np.ndarray:
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance_fn(i, j))
            if not np.isfinite(d):
                raise FloatingPointError(f"non-finite pairwise distance for clients at positions {i},{j}: {d}")
            D[i, j] = D[j, i] = max(d, 0.0)
    return D


def _local_scale_distance(D: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n <= 1:
        return D.copy(), np.ones(n, dtype=np.float64)

    k = int(max(1, min(k, n - 1)))
    sigmas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = np.delete(D[i], i)
        row = np.sort(row[np.isfinite(row)])
        if row.size == 0:
            sigmas[i] = 1.0
        else:
            sigmas[i] = max(float(row[min(k - 1, row.size - 1)]), 1e-8)

    denom = np.sqrt(np.outer(sigmas, sigmas)).clip(min=1e-8)
    scaled = D / denom
    scaled = 0.5 * (scaled + scaled.T)
    np.fill_diagonal(scaled, 0.0)
    return scaled, sigmas


def _layer_weights(layers: Sequence[int]) -> Dict[int, float]:
    """Give somewhat more weight to shallow/mid visual statistics (style-sensitive)."""
    layers = sorted(int(x) for x in layers)
    if not layers:
        return {}
    raw = np.exp(-0.55 * np.arange(len(layers), dtype=np.float64))
    raw /= raw.sum()
    return {layer: float(w) for layer, w in zip(layers, raw)}


def build_multiview_distance(
    signatures: Sequence[Dict],
    prompt_weight: float = 0.20,
    gaussian_weight: float = 0.45,
    quantile_weight: float = 0.35,
    local_scale_k: int = 3,
) -> Dict:
    """Construct a fused, locally-scaled distance matrix from visual client signatures."""
    if len(signatures) == 0:
        raise ValueError("At least one signature is required")

    total_weight = prompt_weight + gaussian_weight + quantile_weight
    if total_weight <= 0:
        raise ValueError("At least one distance weight must be positive")
    prompt_weight /= total_weight
    gaussian_weight /= total_weight
    quantile_weight /= total_weight

    common_layers = set(signatures[0]["layers"].keys())
    for sig in signatures[1:]:
        common_layers &= set(sig["layers"].keys())
    if not common_layers:
        raise ValueError("Signatures have no common visual layers")
    common_layers = sorted(int(x) for x in common_layers)
    lw = _layer_weights(common_layers)
    n = len(signatures)

    prompt_D = _pairwise_view(
        n,
        lambda i, j: prompt_update_angle_distance(
            signatures[i]["prompt_delta"],
            signatures[j]["prompt_delta"],
            signatures[i].get("prompt"),
            signatures[j].get("prompt"),
        ),
    )

    gaussian_D = np.zeros((n, n), dtype=np.float64)
    quantile_D = np.zeros((n, n), dtype=np.float64)

    per_layer_gaussian = {}
    per_layer_quantile = {}

    for layer in common_layers:
        prepared = [
            _prepare_gaussian(
                sig["layers"][layer]["mean"],
                sig["layers"][layer]["cov"],
            )
            for sig in signatures
        ]
        gD = _pairwise_view(
            n,
            lambda i, j, prepared=prepared: _bures_gaussian_distance_prepared(
                prepared[i], prepared[j]
            ),
        )
        qD = _pairwise_view(
            n,
            lambda i, j, layer=layer: projected_quantile_distance(
                signatures[i]["layers"][layer]["quantiles"],
                signatures[j]["layers"][layer]["quantiles"],
            ),
        )

        gD_scaled = _robust_scale_distance_matrix(gD)
        qD_scaled = _robust_scale_distance_matrix(qD)
        gaussian_D += lw[layer] * gD_scaled
        quantile_D += lw[layer] * qD_scaled
        per_layer_gaussian[layer] = gD
        per_layer_quantile[layer] = qD

    prompt_scaled = _robust_scale_distance_matrix(prompt_D)
    gaussian_scaled = _robust_scale_distance_matrix(gaussian_D)
    quantile_scaled = _robust_scale_distance_matrix(quantile_D)

    fused = (
        prompt_weight * prompt_scaled
        + gaussian_weight * gaussian_scaled
        + quantile_weight * quantile_scaled
    )
    fused = 0.5 * (fused + fused.T)
    np.fill_diagonal(fused, 0.0)

    locally_scaled, local_sigmas = _local_scale_distance(fused, k=local_scale_k)

    return {
        "fused_distance": fused,
        "local_scaled_distance": locally_scaled,
        "local_sigmas": local_sigmas,
        "prompt_distance": prompt_D,
        "gaussian_distance": gaussian_D,
        "quantile_distance": quantile_D,
        "per_layer_gaussian": per_layer_gaussian,
        "per_layer_quantile": per_layer_quantile,
        "layer_weights": lw,
        "distance_weights": {
            "prompt": float(prompt_weight),
            "gaussian": float(gaussian_weight),
            "quantile": float(quantile_weight),
        },
    }


def _relabel_noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    """Every client remains addressable: HDBSCAN noise becomes a singleton cluster."""
    labels = np.asarray(labels, dtype=int).copy()
    non_noise = labels[labels >= 0]
    next_label = int(non_noise.max() + 1) if non_noise.size else 0
    for i in range(len(labels)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1

    unique = sorted(set(labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(x)] for x in labels], dtype=int)


def _cluster_hdbscan_or_optics(
    D: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 2,
    cluster_selection_method: str = "eom",
) -> Tuple[np.ndarray, np.ndarray, str]:
    n = D.shape[0]

    if n >= 2:
        upper = D[np.triu_indices(n, k=1)]
        if upper.size and float(np.max(np.abs(upper))) <= 1e-10:
            return np.zeros(n, dtype=int), np.ones(n, dtype=np.float64), "degenerate-single-cluster"

    if n == 1:
        return np.array([0], dtype=int), np.array([1.0]), "single-client"
    if n == 2:

        if D[0, 1] <= 1.0:
            return np.array([0, 0], dtype=int), np.ones(2), "two-client-local-scale"
        return np.array([0, 1], dtype=int), np.ones(2), "two-client-local-scale"

    min_cluster_size = max(2, min(int(min_cluster_size), n))
    min_samples = max(1, min(int(min_samples), n - 1))

    try:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=True,
            copy=True,
        )
        raw_labels = model.fit_predict(D)
        probabilities = getattr(model, "probabilities_", np.ones(n, dtype=np.float64))
        return raw_labels.astype(int), np.asarray(probabilities, dtype=np.float64), "sklearn.HDBSCAN"
    except (ImportError, TypeError, ValueError):
        pass

    try:
        import hdbscan  # type: ignore

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=True,
        )
        raw_labels = model.fit_predict(D)
        probabilities = getattr(model, "probabilities_", np.ones(n, dtype=np.float64))
        return raw_labels.astype(int), np.asarray(probabilities, dtype=np.float64), "hdbscan.HDBSCAN"
    except (ImportError, TypeError, ValueError):
        pass

    from sklearn.cluster import OPTICS

    model = OPTICS(
        min_samples=min_samples,
        metric="precomputed",
        cluster_method="xi",
        xi=0.05,
        min_cluster_size=min_cluster_size,
    )
    raw_labels = model.fit_predict(D)
    probabilities = np.ones(n, dtype=np.float64)
    return raw_labels.astype(int), probabilities, "sklearn.OPTICS"


def _cluster_medoids(labels: np.ndarray, D: np.ndarray, client_ids: Sequence[int]) -> Dict[int, int]:
    medoids = {}
    for c in sorted(set(int(x) for x in labels)):
        members = np.where(labels == c)[0]
        sub = D[np.ix_(members, members)]
        medoid_pos = members[int(np.argmin(sub.mean(axis=1)))]
        medoids[c] = int(client_ids[medoid_pos])
    return medoids


def cluster_visual_signatures(
    signatures: Sequence[Dict],
    client_ids: Optional[Sequence[int]] = None,
    prompt_weight: float = 0.20,
    gaussian_weight: float = 0.45,
    quantile_weight: float = 0.35,
    local_scale_k: int = 3,
    min_cluster_size: int = 2,
    min_samples: int = 2,
    cluster_selection_method: str = "eom",
) -> Dict:
    """Automatically discover client clusters from visual signatures, without K."""
    if client_ids is None:
        client_ids = [int(sig.get("client_id", i)) for i, sig in enumerate(signatures)]
    client_ids = [int(x) for x in client_ids]
    if len(client_ids) != len(signatures):
        raise ValueError("client_ids/signatures length mismatch")

    dist = build_multiview_distance(
        signatures,
        prompt_weight=prompt_weight,
        gaussian_weight=gaussian_weight,
        quantile_weight=quantile_weight,
        local_scale_k=local_scale_k,
    )
    D = dist["local_scaled_distance"]

    raw_labels, probabilities, algorithm = _cluster_hdbscan_or_optics(
        D,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    labels = _relabel_noise_as_singletons(raw_labels)
    medoids = _cluster_medoids(labels, D, client_ids)

    clusters = {}
    for cid, label in zip(client_ids, labels):
        clusters.setdefault(int(label), []).append(int(cid))

    return {
        "client_ids": client_ids,
        "labels": labels,
        "raw_labels": raw_labels,
        "probabilities": probabilities,
        "clusters": clusters,
        "medoids": medoids,
        "algorithm": algorithm,
        **dist,
    }


def ema_signature(previous: Optional[Dict], current: Dict, momentum: float = 0.60) -> Dict:
    """Temporal EMA for signatures to reduce round-to-round cluster jitter.

    A convex combination preserves PSD covariance matrices. Metadata is taken from the
    current signature. This operates only on already-derived visual statistics at server.
    """
    if previous is None or momentum <= 0:
        return current
    if momentum >= 1:
        return previous

    m = float(momentum)
    out = {
        "client_id": int(current["client_id"]),
        "quantile_levels": tuple(current["quantile_levels"]),
        "signature_dim": int(current["signature_dim"]),
    }

    for key in ("prompt", "prompt_delta"):
        a = _as_numpy(previous[key])
        b = _as_numpy(current[key])
        out[key] = torch.from_numpy((m * a + (1.0 - m) * b).astype(np.float32))

    out["layers"] = {}
    for layer, cur_stats in current["layers"].items():
        layer = int(layer)
        if layer not in previous.get("layers", {}):
            out["layers"][layer] = cur_stats
            continue
        prev_stats = previous["layers"][layer]
        out["layers"][layer] = {}
        for key in ("mean", "cov", "quantiles"):
            a = _as_numpy(prev_stats[key])
            b = _as_numpy(cur_stats[key])
            value = m * a + (1.0 - m) * b
            if key == "cov":
                value = 0.5 * (value + value.T)
                value += 1e-6 * np.eye(value.shape[0])
            out["layers"][layer][key] = torch.from_numpy(value.astype(np.float32))
        out["layers"][layer]["n_tokens"] = int(cur_stats["n_tokens"])

    return out


def save_cluster_diagnostics(result: Dict, output_dir: str, round_idx: int) -> None:
    """Persist cluster assignments and numeric distance matrices for reproducible analysis."""
    cluster_dir = os.path.join(output_dir, "visual_clusters")
    os.makedirs(cluster_dir, exist_ok=True)

    json_path = os.path.join(cluster_dir, f"round_{round_idx:03d}.json")
    payload = {
        "round": int(round_idx),
        "algorithm": result["algorithm"],
        "client_ids": [int(x) for x in result["client_ids"]],
        "labels": [int(x) for x in result["labels"].tolist()],
        "raw_labels": [int(x) for x in result["raw_labels"].tolist()],
        "probabilities": [float(x) for x in result["probabilities"].tolist()],
        "clusters": {str(k): [int(x) for x in v] for k, v in result["clusters"].items()},
        "medoids": {str(k): int(v) for k, v in result["medoids"].items()},
        "layer_weights": {str(k): float(v) for k, v in result["layer_weights"].items()},
        "distance_weights": result["distance_weights"],
        "local_sigmas": [float(x) for x in result["local_sigmas"].tolist()],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        os.path.join(cluster_dir, f"round_{round_idx:03d}_distances.npz"),
        client_ids=np.asarray(result["client_ids"], dtype=int),
        fused=result["fused_distance"],
        local_scaled=result["local_scaled_distance"],
        prompt=result["prompt_distance"],
        gaussian=result["gaussian_distance"],
        quantile=result["quantile_distance"],
        labels=result["labels"],
        raw_labels=result["raw_labels"],
    )
