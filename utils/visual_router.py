from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.visual_cluster_utils import build_multiview_distance

_EPS = 1e-12


@dataclass
class RouterConfig:
    topk: int = 3
    domain_weight: float = 0.45
    utility_weight: float = 0.35
    alignment_weight: float = 0.20
    domain_temperature: float = 1.0
    utility_temperature: float = 0.02
    softmax_temperature: float = 0.20
    self_prior: float = 0.65
    min_domain_affinity: float = 0.10
    min_update_cosine: float = -0.10
    reject_negative_utility: bool = True
    utility_reject_margin: float = 0.0
    peer_norm_cap_ratio: float = 1.00
    fusion_strength: float = 0.25
    max_fused_norm_ratio: float = 0.35
    staleness_tau: float = 3.0
    min_cluster_confidence: float = 0.05
    conflict_projection: bool = True

    def validate(self) -> None:
        if self.topk < 1:
            raise ValueError("router topk must be >= 1")
        if self.domain_temperature <= 0 or self.utility_temperature <= 0 or self.softmax_temperature <= 0:
            raise ValueError("router temperatures must be > 0")
        if self.fusion_strength < 0 or self.max_fused_norm_ratio < 0 or self.peer_norm_cap_ratio <= 0:
            raise ValueError("router norm/fusion controls are invalid")
        w = self.domain_weight + self.utility_weight + self.alignment_weight
        if w <= 0:
            raise ValueError("at least one router evidence weight must be positive")


def _tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().float().cpu()
    return torch.as_tensor(x, dtype=torch.float32)


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.float()).item())


def update_cosine(local_update: torch.Tensor, peer_update: torch.Tensor) -> float:
    """Blend global and token-wise cosine for position-aligned visual prompt updates."""
    a = _tensor(local_update)
    b = _tensor(peer_update)
    if a.shape != b.shape:
        raise ValueError(f"update shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")

    af, bf = a.reshape(-1), b.reshape(-1)
    if _norm(af) < 1e-12 or _norm(bf) < 1e-12:
        return 0.0
    global_cos = float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0), dim=1).item())

    if a.ndim >= 2:
        aa = a.reshape(a.shape[0], -1)
        bb = b.reshape(b.shape[0], -1)
        token_cos = F.cosine_similarity(aa, bb, dim=1, eps=1e-12)
        token_cos = float(token_cos.mean().item())
        return float(np.clip(0.5 * global_cos + 0.5 * token_cos, -1.0, 1.0))
    return float(np.clip(global_cos, -1.0, 1.0))


def conflict_project_peer(peer_update: torch.Tensor, local_update: torch.Tensor) -> torch.Tensor:
    """PCGrad-style one-sided projection: remove only the component opposing local update."""
    p = _tensor(peer_update)
    l = _tensor(local_update)
    lf = l.reshape(-1)
    pf = p.reshape(-1)
    denom = torch.dot(lf, lf)
    if float(denom) <= 1e-16:
        return p
    dot = torch.dot(pf, lf)
    if float(dot) < 0.0:
        pf = pf - (dot / denom) * lf
    return pf.reshape_as(p)


def norm_cap_peer(peer_update: torch.Tensor, local_update: torch.Tensor, ratio: float) -> torch.Tensor:
    """Do not let one peer update exceed `ratio * ||local_update||`."""
    p = _tensor(peer_update)
    local_norm = _norm(_tensor(local_update))
    peer_norm = _norm(p)
    if local_norm < 1e-12 or peer_norm < 1e-12:
        return torch.zeros_like(p) if local_norm < 1e-12 else p
    cap = ratio * local_norm
    if peer_norm > cap:
        p = p * (cap / peer_norm)
    return p


def _domain_distances(local_signature: Dict, peer_signatures: Sequence[Dict], distance_kwargs: Mapping) -> np.ndarray:
    """Compute local->peer distances jointly so robust per-view scaling is meaningful."""
    signatures = [local_signature] + list(peer_signatures)
    if len(signatures) <= 1:
        return np.zeros(0, dtype=np.float64)
    dist = build_multiview_distance(signatures, **dict(distance_kwargs))
    fused = np.asarray(dist["fused_distance"], dtype=np.float64)[0, 1:]
    local = np.asarray(dist["local_scaled_distance"], dtype=np.float64)[0, 1:]
    return 0.5 * fused + 0.5 * local


def prepare_candidates(
    local_signature: Dict,
    local_update: torch.Tensor,
    peer_packets: Sequence[Dict],
    cfg: RouterConfig,
    distance_kwargs: Mapping,
    current_round: int,
) -> List[Dict]:
    """Cheap stage: domain/profile ranking + update compatibility + reliability.

    peer_packets fields:
      client_id, update, signature, cluster_confidence, update_round
    """
    cfg.validate()
    local_update = _tensor(local_update)
    if _norm(local_update) < 1e-12 or not peer_packets:
        return []

    valid = []
    for packet in peer_packets:
        if packet.get("signature") is None or packet.get("update") is None:
            continue
        if tuple(_tensor(packet["update"]).shape) != tuple(local_update.shape):
            continue
        valid.append(packet)
    if not valid:
        return []

    distances = _domain_distances(
        local_signature,
        [p["signature"] for p in valid],
        distance_kwargs=distance_kwargs,
    )

    candidates = []
    for packet, distance in zip(valid, distances.tolist()):
        peer_update = _tensor(packet["update"])
        cosine = update_cosine(local_update, peer_update)
        domain_affinity = math.exp(-max(float(distance), 0.0) / cfg.domain_temperature)

        cluster_conf = float(packet.get("cluster_confidence", 1.0))
        cluster_conf = float(np.clip(cluster_conf, cfg.min_cluster_confidence, 1.0))
        update_round = int(packet.get("update_round", current_round))
        age = max(int(current_round) - update_round, 0)
        freshness = math.exp(-float(age) / max(cfg.staleness_tau, 1e-6))
        reliability = float(np.clip(cluster_conf * freshness, 1e-6, 1.0))

        prepared = peer_update
        if cfg.conflict_projection:
            prepared = conflict_project_peer(prepared, local_update)
        prepared = norm_cap_peer(prepared, local_update, cfg.peer_norm_cap_ratio)

        if _norm(prepared) < 1e-12:
            continue
        if domain_affinity < cfg.min_domain_affinity:
            continue
        if cosine < cfg.min_update_cosine:
            continue

      
        alignment_affinity = 0.5 * (cosine + 1.0)
        pre_score = (
            cfg.domain_weight * domain_affinity
            + cfg.alignment_weight * alignment_affinity
        ) * reliability

        candidates.append(
            {
                "client_id": int(packet["client_id"]),
                "distance": float(distance),
                "domain_affinity": float(domain_affinity),
                "update_cosine": float(cosine),
                "alignment_affinity": float(alignment_affinity),
                "cluster_confidence": float(cluster_conf),
                "freshness": float(freshness),
                "reliability": float(reliability),
                "prepared_update": prepared,
                "pre_score": float(pre_score),
            }
        )

    candidates.sort(key=lambda x: x["pre_score"], reverse=True)
    return candidates[: min(cfg.topk, len(candidates))]


def _utility_affinity(relative_loss_gain: float, temperature: float) -> float:
    z = float(relative_loss_gain) / temperature
    return float(0.5 * (math.tanh(z) + 1.0))


def finalize_route(
    local_update: torch.Tensor,
    candidates: Sequence[Dict],
    utility_by_client: Mapping[int, float],
    cfg: RouterConfig,
) -> Dict:
    """Fuse top-k peer updates with an explicit self/abstain route and trust region."""
    cfg.validate()
    local_update = _tensor(local_update)
    local_norm = _norm(local_update)
    if local_norm < 1e-12 or not candidates:
        return {
            "self_weight": 1.0,
            "peer_weights": {},
            "fused_peer_delta": torch.zeros_like(local_update),
            "peer_mass": 0.0,
            "candidates": [],
        }

    kept = []
    scores = []
    total_w = cfg.domain_weight + cfg.utility_weight + cfg.alignment_weight
    wd = cfg.domain_weight / total_w
    wu = cfg.utility_weight / total_w
    wa = cfg.alignment_weight / total_w

    for c in candidates:
        cid = int(c["client_id"])
        utility = float(utility_by_client.get(cid, 0.0))
        if not math.isfinite(utility):
            continue
        if cfg.reject_negative_utility and utility < -abs(cfg.utility_reject_margin):
            continue
        utility_aff = _utility_affinity(utility, cfg.utility_temperature)
        quality = (
            wd * float(c["domain_affinity"])
            + wu * utility_aff
            + wa * float(c["alignment_affinity"])
        )
        quality *= float(c["reliability"])
        quality = max(quality, 1e-8)
        score = math.log(quality)

        cc = dict(c)
        cc["relative_loss_gain"] = utility
        cc["utility_affinity"] = utility_aff
        cc["quality"] = quality
        kept.append(cc)
        scores.append(score)

    if not kept:
        return {
            "self_weight": 1.0,
            "peer_weights": {},
            "fused_peer_delta": torch.zeros_like(local_update),
            "peer_mass": 0.0,
            "candidates": [],
        }

    self_score = math.log(max(cfg.self_prior, 1e-8))
    logits = torch.tensor([self_score] + scores, dtype=torch.float64)
    probs = torch.softmax(logits / cfg.softmax_temperature, dim=0).float()
    self_weight = float(probs[0].item())

    fused = torch.zeros_like(local_update)
    peer_weights: Dict[int, float] = {}
    for prob, c in zip(probs[1:], kept):
        w = float(prob.item())
        peer_weights[int(c["client_id"])] = w
        fused += w * _tensor(c["prepared_update"])

    fused *= float(cfg.fusion_strength)
    max_norm = float(cfg.max_fused_norm_ratio) * local_norm
    fused_norm = _norm(fused)
    if max_norm == 0.0:
        fused.zero_()
        fused_norm = 0.0
    elif fused_norm > max_norm:
        fused *= max_norm / fused_norm
        fused_norm = max_norm

    return {
        "self_weight": self_weight,
        "peer_weights": peer_weights,
        "fused_peer_delta": fused,
        "peer_mass": float(sum(peer_weights.values())),
        "fused_norm": float(fused_norm),
        "local_update_norm": float(local_norm),
        "candidates": kept,
    }
