import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer

from utils.fed_utils import count_parameters
from utils.visual_cluster_utils import (
    cluster_visual_signatures,
    ema_signature,
    save_cluster_diagnostics,
)
from utils.visual_router import (
    RouterConfig,
    prepare_candidates,
    finalize_route,
)


def _parse_int_tuple(text):
    if isinstance(text, (list, tuple)):
        return tuple(int(x) for x in text)
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_float_tuple(text):
    if isinstance(text, (list, tuple)):
        return tuple(float(x) for x in text)
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="pFedVCluster",
        choices=["pFedVCluster"],
        help="visual-prompt client profiling + no-K server clustering; no averaging/gating",
    )

    parser.add_argument("--trainer", type=str, default="PFEDVCLUSTERS")

    parser.add_argument("--round", type=int, default=25)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--num_users", type=int, default=20)
    parser.add_argument("--frac", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="optional visual-prompt gradient norm cap; 0 disables")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu_threads", type=int, default=1, help="CPU/BLAS threads; small profile matrices benefit from one thread")
    parser.add_argument("--prec", type=str, default="amp", choices=["fp16", "fp32", "amp"])


    parser.add_argument("--iid", type=_parse_bool, default=False)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--useall", type=_parse_bool, default=True)
    parser.add_argument("--partition", type=str, default="noniid-labeldir100")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--imbalance_train", type=_parse_bool, default=True)
    parser.add_argument("--split_client", type=_parse_bool, default=True)
    parser.add_argument("--num_domain", type=int, default=4)

    parser.add_argument("--n_ctx", type=int, default=16, help="number of trainable VISUAL prompt tokens")
    parser.add_argument("--visual_prompt_dropout", type=float, default=0.0)
    parser.add_argument(
        "--signature_layers",
        type=str,
        default="2,5,8",
        help="1-based CLIP ViT blocks used for multi-scale patch-token profiling",
    )
    parser.add_argument("--signature_dim", type=int, default=48)
    parser.add_argument("--signature_batches", type=int, default=8)
    parser.add_argument("--signature_tokens_per_batch", type=int, default=512)
    parser.add_argument("--signature_quantiles", type=str, default="0.10,0.25,0.50,0.75,0.90")
    parser.add_argument("--signature_proj_seed", type=int, default=2026)
    parser.add_argument(
        "--signature_momentum",
        type=float,
        default=0.60,
        help="server EMA for client visual profiles; 0 disables temporal smoothing",
    )

    # Multi-view server distance. Token-distribution views dominate prompt direction.
    parser.add_argument("--prompt_distance_weight", type=float, default=0.20)
    parser.add_argument("--gaussian_distance_weight", type=float, default=0.45)
    parser.add_argument("--quantile_distance_weight", type=float, default=0.35)
    parser.add_argument("--local_scale_k", type=int, default=3)

    parser.add_argument("--cluster_min_size", type=int, default=2)
    parser.add_argument("--cluster_min_samples", type=int, default=2)
    parser.add_argument(
        "--cluster_selection_method",
        type=str,
        default="eom",
        choices=["eom", "leaf"],
    )
    # Lightweight CLIENT-SIDE router. No trainable attention/gating network is introduced.
    parser.add_argument("--router_topk", type=int, default=3, help="max peers probed per client/round")
    parser.add_argument("--router_calibration_batches", type=int, default=1)
    parser.add_argument("--router_probe_scale", type=float, default=0.20)
    parser.add_argument("--router_domain_weight", type=float, default=0.45)
    parser.add_argument("--router_utility_weight", type=float, default=0.35)
    parser.add_argument("--router_alignment_weight", type=float, default=0.20)
    parser.add_argument("--router_domain_temperature", type=float, default=1.0)
    parser.add_argument("--router_utility_temperature", type=float, default=0.02)
    parser.add_argument("--router_softmax_temperature", type=float, default=0.20)
    parser.add_argument("--router_self_prior", type=float, default=0.65)
    parser.add_argument("--router_min_domain_affinity", type=float, default=0.10)
    parser.add_argument("--router_min_update_cosine", type=float, default=-0.10)
    parser.add_argument("--router_utility_reject_margin", type=float, default=0.0)
    parser.add_argument("--router_peer_norm_cap_ratio", type=float, default=1.0)
    parser.add_argument("--router_fusion_strength", type=float, default=0.25)
    parser.add_argument("--router_max_fused_norm_ratio", type=float, default=0.35)
    parser.add_argument("--router_staleness_tau", type=float, default=3.0)
    parser.add_argument("--router_disable_functional_probe", action="store_true")
    parser.add_argument("--router_disable_conflict_projection", action="store_true")
    parser.add_argument("--router_allow_negative_utility", action="store_true")

    # Paths/configs.
    parser.add_argument("--logdir", type=str, default="./logs/")
    parser.add_argument("--root", type=str, default="./dataset/")
    parser.add_argument("--output-dir", type=str, default="./output/office/pFedVCluster/vit_b16_router_checked")
    parser.add_argument("--config-file", type=str, default="./configs/trainers/pFedVClusters/vit_b16.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="./configs/datasets/office.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    for key in sorted(args.__dict__.keys()):
        print(f"{key}: {getattr(args, key)}")
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def extend_cfg(cfg, args):
    from yacs.config import CfgNode as CN

    # Baseline namespace retained because other parts of the pFedVClusters repo may expect it.
    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx
    cfg.TRAINER.PROMPTFL.CSC = False
    cfg.TRAINER.PROMPTFL.CTX_INIT = False
    cfg.TRAINER.PROMPTFL.PREC = args.prec
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.PFEDVCLUSTERS = CN()
    cfg.TRAINER.PFEDVCLUSTERS.GRAD_CLIP = args.grad_clip
    cfg.TRAINER.PFEDVCLUSTERS.N_CTX = args.n_ctx
    cfg.TRAINER.PFEDVCLUSTERS.PREC = args.prec
    cfg.TRAINER.PFEDVCLUSTERS.VISUAL_PROMPT_DROPOUT = args.visual_prompt_dropout
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_LAYERS = _parse_int_tuple(args.signature_layers)
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_DIM = args.signature_dim
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_BATCHES = args.signature_batches
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_TOKENS_PER_BATCH = args.signature_tokens_per_batch
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_QUANTILES = _parse_float_tuple(args.signature_quantiles)
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_PROJ_SEED = args.signature_proj_seed


    cfg.TRAINER.PFEDVCLUSTERS.CSC = False
    cfg.TRAINER.PFEDVCLUSTERS.CTX_INIT = False
    cfg.TRAINER.PFEDVCLUSTERS.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PFEDVCLUSTERS.NUM_EXPERTS = 1
    cfg.TRAINER.PFEDVCLUSTERS.GATING_HEADS = 1
    cfg.TRAINER.PFEDVCLUSTERS.GATING_EMBED_DIM = 1
    cfg.TRAINER.PFEDVCLUSTERS.LMBDA = 0.0
    cfg.TRAINER.PFEDVCLUSTERS.SCALING = 1.0

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.USERS = args.num_users
    cfg.DATASET.IID = args.iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train
    cfg.DATASET.SPLIT_CLIENT = args.split_client

    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain
    cfg.OPTIM.ROUND = args.round
    cfg.OPTIM.MAX_EPOCH = args.local_epochs
    cfg.OPTIM.GAMMA = args.gamma
    if args.lr is not None:
        cfg.OPTIM.LR = args.lr
    cfg.MODEL.BACKBONE.PRETRAINED = True
    cfg.TEST.NO_TEST = True


def reset_cfg(cfg, args):
    if args.lr is not None:
        cfg.OPTIM.LR = args.lr
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.round:
        cfg.OPTIM.ROUND = args.round
    if args.local_epochs:
        cfg.OPTIM.MAX_EPOCH = args.local_epochs
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.TRAINER.PFEDVCLUSTERS.GRAD_CLIP = args.grad_clip
    cfg.TRAINER.PFEDVCLUSTERS.N_CTX = args.n_ctx
    cfg.TRAINER.PFEDVCLUSTERS.PREC = args.prec
    cfg.TRAINER.PFEDVCLUSTERS.VISUAL_PROMPT_DROPOUT = args.visual_prompt_dropout
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_LAYERS = _parse_int_tuple(args.signature_layers)
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_DIM = args.signature_dim
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_BATCHES = args.signature_batches
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_TOKENS_PER_BATCH = args.signature_tokens_per_batch
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_QUANTILES = _parse_float_tuple(args.signature_quantiles)
    cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_PROJ_SEED = args.signature_proj_seed


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    cfg.set_new_allowed(True)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def _evaluate(results, epoch, max_epoch, args, cfg, metric_history, start_time):
    if any(r is None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        raise RuntimeError(f"Missing test results for clients {missing}")

    acc = [float(r[0]) for r in results]
    err = [float(r[1]) for r in results]
    f1 = [float(r[2]) for r in results]

    mean_acc = float(np.mean(acc))
    mean_err = float(np.mean(err))
    mean_f1 = float(np.mean(f1))
    metric_history["epoch"].append(epoch)
    metric_history["time"].append(time.time() - start_time)
    metric_history["acc"].append(mean_acc)
    metric_history["err"].append(mean_err)
    metric_history["f1"].append(mean_f1)
    metric_history.setdefault("client_acc", []).append(acc)

    print("Global mean personalized test acc:", mean_acc)
    print("Global mean personalized test error:", mean_err)
    print("Global mean personalized macro_f1:", mean_f1)

    dataset_name = str(cfg.DATASET.NAME)
    domain_map = {
        "DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        "Office": ["amazon", "caltech", "dslr", "webcam"],
        "OfficeCaltech10": ["amazon", "caltech", "dslr", "webcam"],
    }
    if dataset_name in domain_map and args.split_client:
        domains = domain_map[dataset_name]
        per_domain = args.num_users // len(domains)
        if per_domain > 0:
            print("Test acc of clients:", acc)
            for i, domain in enumerate(domains):
                values = acc[i * per_domain : (i + 1) * per_domain]
                if values:
                    print(f"Test acc of {domain}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            print(f"Test acc of all: {np.mean(acc):.4f} ± {np.std(acc):.4f}")

    print("------------local test finish-------------")


def _report_domain_cluster_quality(cluster_result, args, cfg):
    """Evaluation-only ARI/NMI against known benchmark domains; never used by clustering."""
    domain_counts = {"Office": 4, "OfficeCaltech10": 4, "DomainNet": 6}
    dataset_name = str(cfg.DATASET.NAME)
    if dataset_name not in domain_counts or not args.split_client:
        return
    n_domains = domain_counts[dataset_name]
    if args.num_users % n_domains != 0:
        return
    per_domain = args.num_users // n_domains
    client_ids = [int(x) for x in cluster_result["client_ids"]]
    if len(client_ids) < 2:
        return
    true_domains = [min(cid // per_domain, n_domains - 1) for cid in client_ids]
    predicted = [int(x) for x in cluster_result["labels"]]
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(true_domains, predicted)
        nmi = normalized_mutual_info_score(true_domains, predicted)
        print(f"[visual-clustering/eval-only] domain ARI={ari:.4f} NMI={nmi:.4f}")
    except ImportError:
        pass


def main(args):
    if args.resume or args.eval_only or args.no_train:
        raise ValueError("Federated resume/eval-only/no-train is not implemented; refusing to silently train.")
    if not 0 < args.frac <= 1 or args.round < 1 or args.num_users < 1:
        raise ValueError("Require 0 < frac <= 1 and positive rounds/client count")
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print_args(args, cfg)

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model, "visual_prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")

    n_clients = cfg.DATASET.USERS
    initial_prompt = local_trainer.get_initial_visual_prompt()
    initial_optim_state = local_trainer.get_client_optim_state()

    # Persistent PERSONALIZED state. Peer fusion is deliberately kept separate.
    local_visual_prompts = [None for _ in range(n_clients)]
    local_optim_states = [None for _ in range(n_clients)]
    local_prompt_updates = [None for _ in range(n_clients)]
    update_last_round = [-1 for _ in range(n_clients)]

    # Server-side profile state. There is no server prompt/model to average.
    server_signature_bank = [None for _ in range(n_clients)]
    signature_last_round = [-1 for _ in range(n_clients)]
    cluster_labels = np.full(n_clients, -1, dtype=int)
    cluster_confidence = np.zeros(n_clients, dtype=np.float64)

    # Client-side generalization residual for the next train/test pass.
    routed_peer_deltas = [None for _ in range(n_clients)]
    router_diagnostics = [None for _ in range(n_clients)]

    router_cfg = RouterConfig(
        topk=args.router_topk,
        domain_weight=args.router_domain_weight,
        utility_weight=args.router_utility_weight,
        alignment_weight=args.router_alignment_weight,
        domain_temperature=args.router_domain_temperature,
        utility_temperature=args.router_utility_temperature,
        softmax_temperature=args.router_softmax_temperature,
        self_prior=args.router_self_prior,
        min_domain_affinity=args.router_min_domain_affinity,
        min_update_cosine=args.router_min_update_cosine,
        reject_negative_utility=not args.router_allow_negative_utility,
        utility_reject_margin=args.router_utility_reject_margin,
        peer_norm_cap_ratio=args.router_peer_norm_cap_ratio,
        fusion_strength=args.router_fusion_strength,
        max_fused_norm_ratio=args.router_max_fused_norm_ratio,
        staleness_tau=args.router_staleness_tau,
        conflict_projection=not args.router_disable_conflict_projection,
    )
    router_cfg.validate()

    distance_kwargs = dict(
        prompt_weight=args.prompt_distance_weight,
        gaussian_weight=args.gaussian_distance_weight,
        quantile_weight=args.quantile_distance_weight,
        local_scale_k=args.local_scale_k,
    )

    max_epoch = cfg.OPTIM.ROUND
    history = {"epoch": [], "time": [], "acc": [], "err": [], "f1": []}
    start = time.time()
    participation_rng = np.random.default_rng(max(0, cfg.SEED))

    for epoch in range(max_epoch):
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = participation_rng.choice(range(args.num_users), m, replace=False)
        idxs_users = [int(x) for x in idxs_users]
        print("idxs_users", idxs_users)
        print("------------local train start epoch:", epoch, "-------------")

        for idx in idxs_users:
            # Resume the client's own private prompt/optimizer trajectory.
            if local_visual_prompts[idx] is None:
                local_trainer.load_visual_prompt(initial_prompt)
                local_trainer.load_client_optim_state(initial_optim_state)
            else:
                local_trainer.load_visual_prompt(local_visual_prompts[idx])
                local_trainer.load_client_optim_state(local_optim_states[idx])

            # Add the previously routed generalization residual only in the forward path.
            local_trainer.set_peer_generalization_delta(routed_peer_deltas[idx])
            before_prompt = local_trainer.get_visual_prompt()
            local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
            after_prompt = local_trainer.get_visual_prompt()

            local_visual_prompts[idx] = after_prompt
            local_optim_states[idx] = local_trainer.get_client_optim_state()
            local_prompt_updates[idx] = (after_prompt - before_prompt).detach().float().cpu()
            update_last_round[idx] = epoch

            # Signature intentionally ignores routed peer residual: clustering must measure the
            # client's intrinsic visual domain, not recursively reinforce last round's route.
            current_signature = local_trainer.build_visual_signature(idx)
            server_signature_bank[idx] = ema_signature(
                server_signature_bank[idx],
                current_signature,
                momentum=args.signature_momentum,
            )
            signature_last_round[idx] = epoch

        print("------------local train finish epoch:", epoch, "-------------")

        # ---------------------- SERVER: CLUSTER ONLY, NEVER FUSE ----------------------
        profiled_ids = [i for i, sig in enumerate(server_signature_bank) if sig is not None]
        cluster_result = None
        if len(profiled_ids) >= 2:
            signatures = [server_signature_bank[i] for i in profiled_ids]
            cluster_result = cluster_visual_signatures(
                signatures,
                client_ids=profiled_ids,
                **distance_kwargs,
                min_cluster_size=args.cluster_min_size,
                min_samples=args.cluster_min_samples,
                cluster_selection_method=args.cluster_selection_method,
            )
            cluster_labels[:] = -1
            cluster_confidence[:] = 0.0
            for pos, (cid, label) in enumerate(zip(cluster_result["client_ids"], cluster_result["labels"])):
                cluster_labels[int(cid)] = int(label)
                cluster_confidence[int(cid)] = float(cluster_result["probabilities"][pos])

            save_cluster_diagnostics(cluster_result, cfg.OUTPUT_DIR, epoch)
            print(f"[visual-clustering] round={epoch} algorithm={cluster_result['algorithm']}")
            print(f"[visual-clustering] clusters={cluster_result['clusters']}")
            _report_domain_cluster_quality(cluster_result, args, cfg)
            profile_age = {cid: epoch - signature_last_round[cid] for cid in profiled_ids}
            print(f"[visual-clustering] profile_age={profile_age}")
        else:
            print(f"[visual-clustering] round={epoch}: need >=2 profiled clients; have {len(profiled_ids)}")

        # -------------------- CLIENT: DOMAIN-SENSITIVE PEER ROUTER --------------------
        # This loop is client-side logic executed sequentially in this single-process simulator.
        # The server only supplies same-cluster packets; weights/fusion are never computed server-side.
        if cluster_result is not None and router_cfg.fusion_strength > 0:
            cluster_members = {}
            for cid in profiled_ids:
                label = int(cluster_labels[cid])
                cluster_members.setdefault(label, []).append(cid)

            for idx in profiled_ids:
                if local_visual_prompts[idx] is None or local_prompt_updates[idx] is None:
                    routed_peer_deltas[idx] = None
                    router_diagnostics[idx] = None
                    continue

                members = cluster_members.get(int(cluster_labels[idx]), [])
                peer_packets = []
                for peer_id in members:
                    if peer_id == idx:
                        continue
                    if local_prompt_updates[peer_id] is None or server_signature_bank[peer_id] is None:
                        continue
                    peer_packets.append(
                        {
                            "client_id": int(peer_id),
                            "update": local_prompt_updates[peer_id],
                            "signature": server_signature_bank[peer_id],
                            "cluster_confidence": float(cluster_confidence[peer_id]),
                            "update_round": int(update_last_round[peer_id]),
                        }
                    )

                candidates = prepare_candidates(
                    local_signature=server_signature_bank[idx],
                    local_update=local_prompt_updates[idx],
                    peer_packets=peer_packets,
                    cfg=router_cfg,
                    distance_kwargs=distance_kwargs,
                    current_round=epoch,
                )

                # Load this client's PRIVATE prompt for image-level functional routing.
                local_trainer.load_visual_prompt(local_visual_prompts[idx])
                local_trainer.set_peer_generalization_delta(None)
                candidate_deltas = {int(c["client_id"]): c["prepared_update"] for c in candidates}

                if candidates and not args.router_disable_functional_probe:
                    probe = local_trainer.probe_peer_update_utilities(
                        idx=idx,
                        candidate_deltas=candidate_deltas,
                        calibration_batches=args.router_calibration_batches,
                        probe_scale=args.router_probe_scale,
                    )
                    utilities = probe["utility"]
                    base_probe_loss = probe["base_loss"]
                else:
                    utilities = {cid: 0.0 for cid in candidate_deltas}
                    base_probe_loss = None

                route = finalize_route(
                    local_update=local_prompt_updates[idx],
                    candidates=candidates,
                    utility_by_client=utilities,
                    cfg=router_cfg,
                )
                # Probe the actual weighted residual too: helpful individual updates
                # need not be helpful when combined. This uses training images only.
                fused_gain = None
                if route["peer_mass"] > 0 and not args.router_disable_functional_probe:
                    check = local_trainer.probe_peer_update_utilities(
                        idx, {-1: route["fused_peer_delta"]},
                        calibration_batches=args.router_calibration_batches, probe_scale=1.0,
                    )
                    fused_gain = check["utility"][-1]
                    if not np.isfinite(fused_gain) or fused_gain <= 0:
                        route.update(fused_peer_delta=torch.zeros_like(local_prompt_updates[idx]),
                                     self_weight=1.0, peer_weights={}, peer_mass=0.0, fused_norm=0.0)
                routed_peer_deltas[idx] = route["fused_peer_delta"].detach().float().cpu()
                router_diagnostics[idx] = {
                    "round": int(epoch),
                    "cluster": int(cluster_labels[idx]),
                    "base_probe_loss": base_probe_loss,
                    "fused_relative_loss_gain": fused_gain,
                    "self_weight": float(route["self_weight"]),
                    "peer_weights": {int(k): float(v) for k, v in route["peer_weights"].items()},
                    "peer_mass": float(route["peer_mass"]),
                    "fused_norm": float(route.get("fused_norm", 0.0)),
                    "local_update_norm": float(route.get("local_update_norm", 0.0)),
                    "candidates": [
                        {
                            "client_id": int(c["client_id"]),
                            "domain_affinity": float(c["domain_affinity"]),
                            "update_cosine": float(c["update_cosine"]),
                            "relative_loss_gain": float(c.get("relative_loss_gain", 0.0)),
                            "reliability": float(c["reliability"]),
                        }
                        for c in route.get("candidates", [])
                    ],
                }
                print(
                    f"[visual-router] client={idx} cluster={cluster_labels[idx]} "
                    f"self={route['self_weight']:.3f} peers={route['peer_weights']} "
                    f"peer_mass={route['peer_mass']:.3f}"
                )

        # -------------------------- PERSONALIZED + GENERALIZED TEST ------------------
        print("------------local test start-------------")
        results = [None for _ in range(n_clients)]
        for idx in range(n_clients):
            if local_visual_prompts[idx] is None:
                local_trainer.load_visual_prompt(initial_prompt)
                local_trainer.set_peer_generalization_delta(None)
            else:
                local_trainer.load_visual_prompt(local_visual_prompts[idx])
                local_trainer.set_peer_generalization_delta(routed_peer_deltas[idx])
            results[idx] = local_trainer.test(idx=idx)

        _evaluate(results, epoch, max_epoch, args, cfg, history, start)
        print("Round on server :", epoch)
        with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as handle:
            json.dump(history, handle, indent=2)

        torch.save(
            {
                "round": epoch,
                "config": cfg.dump(),
                "args": vars(args),
                "local_optim_states": local_optim_states,
                "history": history,
                "local_visual_prompts": local_visual_prompts,
                "local_prompt_updates": local_prompt_updates,
                "update_last_round": update_last_round,
                "cluster_labels": cluster_labels,
                "cluster_confidence": cluster_confidence,
                "signature_last_round": signature_last_round,
                "routed_peer_deltas": routed_peer_deltas,
                "router_diagnostics": router_diagnostics,
            },
            os.path.join(cfg.OUTPUT_DIR, "visual_prompt_clients_latest.pt"),
        )

    local_trainer.fed_after_train()

    print("global_test_acc_list:", history["acc"])
    print("maximum test acc:", max(history["acc"]) if history["acc"] else None)
    if history["acc"]:
        tail = history["acc"][-5:]
        print("mean of acc:", float(np.mean(tail)))
        print("std of acc:", float(np.std(tail)))
    print("final cluster labels:", cluster_labels.tolist())
    print("final router diagnostics:", router_diagnostics)


if __name__ == "__main__":
    args = get_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        main(args)
