import os.path as osp
import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_checkpoint
from Dassl.dassl.optim import build_optimizer
from utils.training_utils import build_client_scheduler

from clip import clip
from trainers.promptfl import load_clip_to_cpu

_FIXED_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
)


def _safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class VisualPromptLearner(nn.Module):
    """Shallow VPT-style learnable tokens for OpenAI CLIP's VisionTransformer.

    The tokens live in the ViT hidden-width space and are inserted immediately after
    the CLS token, before the frozen Transformer blocks.
    """

    def __init__(self, cfg, image_encoder: nn.Module):
        super().__init__()

        required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
        missing = [name for name in required if not hasattr(image_encoder, name)]
        if missing:
            raise TypeError(
                "PFED visual prompting requires an OpenAI-CLIP VisionTransformer. "
                f"Missing attributes {missing}. Use a ViT backbone such as ViT-B/16, not RN50."
            )

        n_ctx = int(cfg.TRAINER.PFEDVCLUSTERS.N_CTX)
        width = int(image_encoder.conv1.out_channels)
        dtype = image_encoder.conv1.weight.dtype

        if n_ctx <= 0:
            raise ValueError("TRAINER.PFEDVCLUSTERS.N_CTX must be > 0 for visual prompt tuning")

        ctx = torch.empty(n_ctx, width, dtype=dtype)
        # VPT/ViT-style small random initialization.
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)

        self.register_buffer("initial_ctx", ctx.detach().clone(), persistent=True)

        dropout = float(getattr(cfg.TRAINER.PFEDVCLUSTERS, "VISUAL_PROMPT_DROPOUT", 0.0))
        self.dropout = nn.Dropout(dropout)
        self.n_ctx = n_ctx
        self.width = width

    def forward(self, batch_size: int) -> torch.Tensor:
        prompts = self.dropout(self.ctx)
        return prompts.unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def raw_tokens(self) -> torch.Tensor:
        return self.ctx.detach().clone()

    @torch.no_grad()
    def delta_tokens(self) -> torch.Tensor:
        return (self.ctx - self.initial_ctx).detach().clone()


class VisualPromptedCLIP(nn.Module):

    def __init__(self, cfg, classnames: Sequence[str], clip_model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_class = len(self.classnames)

        self.image_encoder = clip_model.visual
        self.visual_prompt_learner = VisualPromptLearner(cfg, self.image_encoder)
        self.register_buffer("peer_generalization_delta", None, persistent=False)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        fixed_text = self._build_fixed_text_features(clip_model, self.classnames)
        self.register_buffer("fixed_text_features", fixed_text, persistent=True)

        signature_dim = int(cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_DIM)
        width = self.visual_prompt_learner.width
        signature_dim = max(2, min(signature_dim, width))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_PROJ_SEED))
        random_matrix = torch.randn(width, signature_dim, generator=g, dtype=torch.float32)
        q, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("signature_projection", q, persistent=True)

    @staticmethod
    @torch.no_grad()
    def _build_fixed_text_features(clip_model: nn.Module, classnames: Sequence[str]) -> torch.Tensor:
        per_template = []
        device = next(clip_model.parameters()).device

        for template in _FIXED_TEMPLATES:
            texts = [template.format(name) for name in classnames]
            tokenized = clip.tokenize(texts).to(device)
            features = clip_model.encode_text(tokenized)
            features = _safe_l2_normalize(features.float())
            per_template.append(features)

        text_features = torch.stack(per_template, dim=0).mean(dim=0)
        return _safe_l2_normalize(text_features).detach()

    @property
    def num_visual_blocks(self) -> int:
        return len(self.image_encoder.transformer.resblocks)

    def _validate_signature_layers(self, collect_layers: Optional[Iterable[int]]) -> Tuple[int, ...]:
        if collect_layers is None:
            return tuple()
        layers = tuple(sorted(set(int(x) for x in collect_layers)))
        for layer in layers:
            if layer < 1 or layer > self.num_visual_blocks:
                raise ValueError(
                    f"Signature layer {layer} is invalid for a {self.num_visual_blocks}-block ViT. "
                    "Layer indices are 1-based."
                )
        return layers

    @torch.no_grad()
    def set_peer_generalization_delta(self, delta_tokens: Optional[torch.Tensor]) -> None:
        if delta_tokens is None:
            self.peer_generalization_delta = None
            return
        expected = tuple(self.visual_prompt_learner.ctx.shape)
        if tuple(delta_tokens.shape) != expected:
            raise ValueError(
                f"peer delta shape mismatch: got {tuple(delta_tokens.shape)}, expected {expected}"
            )
        self.peer_generalization_delta = delta_tokens.detach().to(
            device=self.visual_prompt_learner.ctx.device,
            dtype=self.visual_prompt_learner.ctx.dtype,
        ).clone()

    def encode_image_with_tokens(
        self,
        image: torch.Tensor,
        collect_layers: Optional[Iterable[int]] = None,
        use_peer_guidance: bool = True,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:

        visual = self.image_encoder
        collect_layers = self._validate_signature_layers(collect_layers)

        x = image.type(visual.conv1.weight.dtype)
        x = visual.conv1(x)                          
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)                     

        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)             
        if x.shape[1] != visual.positional_embedding.shape[0]:
            raise RuntimeError(
                "Input resolution/patch grid does not match CLIP positional embeddings: "
                f"tokens={x.shape[1]}, positional={visual.positional_embedding.shape[0]}"
            )
        x = x + visual.positional_embedding.to(x.dtype)

        local_prompts = self.visual_prompt_learner(x.shape[0]).to(dtype=x.dtype, device=x.device)
        prompts = local_prompts
        if use_peer_guidance and self.peer_generalization_delta is not None:
            peer_delta = self.peer_generalization_delta.detach().to(dtype=x.dtype, device=x.device)
            peer_delta = peer_delta.unsqueeze(0).expand(x.shape[0], -1, -1)
            prompts = prompts + peer_delta

        x = torch.cat([x[:, :1, :], prompts, x[:, 1:, :]], dim=1)
        x = visual.ln_pre(x)

        prompt_len = prompts.shape[1]
        patch_start = 1 + prompt_len
        token_layers: Dict[int, torch.Tensor] = {}

        x = x.permute(1, 0, 2)
        for layer_idx, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if layer_idx in collect_layers:
                layer_tokens = x.permute(1, 0, 2)[:, patch_start:, :]
                token_layers[layer_idx] = layer_tokens

        x = x.permute(1, 0, 2)
        cls_feature = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            cls_feature = cls_feature @ visual.proj

        return cls_feature, token_layers

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features, _ = self.encode_image_with_tokens(image, collect_layers=None)
        image_features = _safe_l2_normalize(image_features.float())
        text_features = self.fixed_text_features.float()
        logit_scale = self.logit_scale.float().exp()
        return logit_scale * image_features @ text_features.t()


class PFEDVCLUSTERS(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PFEDVCLUSTERS.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        super().build_data_loader()
        from Dassl.dassl.data.data_manager import build_data_loader
        from Dassl.dassl.data.transforms import build_transform
        import random

        cfg = self.cfg
        tfm = build_transform(cfg, is_train=False)
        self.profile_loaders = {}
        for idx, source in enumerate(self.dm.dataset.federated_train_x):
            records = list(source)
            random.Random(max(0, cfg.SEED) + idx).shuffle(records)
            loader = build_data_loader(
                cfg, sampler_type="SequentialSampler", data_source=records,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, tfm=tfm, is_train=False,
            )
            
            loader.generator = torch.Generator().manual_seed(max(0, cfg.SEED) + idx)
            self.profile_loaders[idx] = loader

    def build_model(self):
        print("==> PFED visual-prompt client model (no gating, no prompt averaging)")
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model.float()
        self.model = VisualPromptedCLIP(cfg, classnames, clip_model)
        if cfg.TRAINER.PFEDVCLUSTERS.PREC == "fp16":
            self.model.half()

        # Freeze absolutely everything, then re-enable only visual prompt tokens.
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.model.visual_prompt_learner.parameters():
            param.requires_grad_(True)

        self.model.to(self.device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"# total params: {total:,}")
        print(f"# trainable visual-prompt params: {trainable:,}")
        print(f"# visual Transformer blocks: {self.model.num_visual_blocks}")

        self.optim_p = build_optimizer(self.model.visual_prompt_learner, cfg.OPTIM)
        self.sched_p = build_client_scheduler(self.optim_p, cfg.OPTIM)
        self.register_model(
            "visual_prompt_learner",
            self.model.visual_prompt_learner,
            self.optim_p,
            self.sched_p,
        )

        self.scaler = GradScaler() if cfg.TRAINER.PFEDVCLUSTERS.PREC == "amp" else None

        self.signature_layers = tuple(int(x) for x in cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_LAYERS)
        self.signature_quantiles = tuple(float(x) for x in cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_QUANTILES)
        self.signature_batches = int(cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_BATCHES)
        self.signature_tokens_per_batch = int(cfg.TRAINER.PFEDVCLUSTERS.SIGNATURE_TOKENS_PER_BATCH)
        self.model._validate_signature_layers(self.signature_layers)

    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PFEDVCLUSTERS.PREC

        if prec == "amp":
            self.optim_p.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim_p)
            if self.cfg.TRAINER.PFEDVCLUSTERS.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.visual_prompt_learner.parameters(),
                                               self.cfg.TRAINER.PFEDVCLUSTERS.GRAD_CLIP)
            self.scaler.step(self.optim_p)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.optim_p.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.TRAINER.PFEDVCLUSTERS.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.visual_prompt_learner.parameters(),
                                               self.cfg.TRAINER.PFEDVCLUSTERS.GRAD_CLIP)
            self.optim_p.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(["visual_prompt_learner"])

        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label

    @torch.no_grad()
    def set_peer_generalization_delta(self, delta_tokens: Optional[torch.Tensor]) -> None:
        """Install/clear this client's frozen routed peer residual."""
        self.model.set_peer_generalization_delta(delta_tokens)

    def get_client_optim_state(self) -> Dict:
        """Snapshot optimizer/scheduler state so sequentially simulated clients stay isolated."""
        return {
            "scaler": copy.deepcopy(self.scaler.state_dict()) if self.scaler is not None else None,
            "optim": copy.deepcopy(self.optim_p.state_dict()),
            "sched": copy.deepcopy(self.sched_p.state_dict()) if self.sched_p is not None else None,
        }

    def load_client_optim_state(self, state: Dict) -> None:
        """Restore the optimizer/scheduler belonging to the current client."""
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(copy.deepcopy(state["scaler"]))
        self.optim_p.load_state_dict(copy.deepcopy(state["optim"]))
        if self.sched_p is not None and state.get("sched") is not None:
            self.sched_p.load_state_dict(copy.deepcopy(state["sched"]))

    @torch.no_grad()
    def get_visual_prompt(self) -> torch.Tensor:
        """Return this client's trained visual prompt tokens on CPU."""
        return self.model.visual_prompt_learner.raw_tokens().float().cpu()

    @torch.no_grad()
    def load_visual_prompt(self, prompt_tokens: torch.Tensor) -> None:
        """Load a client's own visual prompt tokens"""
        if prompt_tokens is None:
            raise ValueError("prompt_tokens cannot be None")
        expected = self.model.visual_prompt_learner.ctx.shape
        if tuple(prompt_tokens.shape) != tuple(expected):
            raise ValueError(f"visual prompt shape mismatch: got {tuple(prompt_tokens.shape)}, expected {tuple(expected)}")
        self.model.visual_prompt_learner.ctx.copy_(
            prompt_tokens.to(
                device=self.model.visual_prompt_learner.ctx.device,
                dtype=self.model.visual_prompt_learner.ctx.dtype,
            )
        )

    @torch.no_grad()
    def get_initial_visual_prompt(self) -> torch.Tensor:
        return self.model.visual_prompt_learner.initial_ctx.detach().float().cpu().clone()

    @staticmethod
    def _deterministic_token_subsample(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """Deterministically subsample [M,D] token rows without RNG-induced signature jitter."""
        if max_tokens <= 0 or tokens.shape[0] <= max_tokens:
            return tokens
        step = max(tokens.shape[0] // max_tokens, 1)
        sampled = tokens[::step]
        return sampled[:max_tokens]

    @torch.no_grad()
    def probe_peer_update_utilities(
        self,
        idx: int,
        candidate_deltas: Dict[int, torch.Tensor],
        calibration_batches: int = 1,
        probe_scale: float = 0.20,
    ) -> Dict:

        if calibration_batches <= 0 or not candidate_deltas:
            return {"base_loss": None, "utility": {int(k): 0.0 for k in candidate_deltas}}

        loader = self.profile_loaders[idx]
        batches = []
        for bidx, batch in enumerate(loader):
            if bidx >= calibration_batches:
                break
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device)
            batches.append((image, label))
        if not batches:
            return {"base_loss": None, "utility": {int(k): 0.0 for k in candidate_deltas}}

        was_training = self.model.training
        previous = None if self.model.peer_generalization_delta is None else self.model.peer_generalization_delta.detach().clone()
        self.model.eval()

        def mean_loss(delta):
            self.model.set_peer_generalization_delta(delta)
            total_loss, count = 0.0, 0
            for image, label in batches:
                logits = self.model(image)
                total_loss += float(F.cross_entropy(logits, label, reduction="sum").item())
                count += int(label.numel())
            return total_loss / count

        try:
            base_loss = mean_loss(None)
            utility = {}
            for cid, delta in candidate_deltas.items():
                candidate_loss = mean_loss(delta.detach() * float(probe_scale))
                utility[int(cid)] = float((base_loss - candidate_loss) / max(abs(base_loss), 1e-8))
            return {"base_loss": float(base_loss), "utility": utility}
        finally:
            self.model.set_peer_generalization_delta(previous)
            self.model.train(was_training)

    @torch.no_grad()
    def build_visual_signature(self, idx: int) -> Dict:
        loader = self.profile_loaders[idx]
        projection = self.model.signature_projection.float()
        proj_dim = projection.shape[1]

        accum = {}
        for layer in self.signature_layers:
            accum[layer] = {
                "n": 0,
                "sum": torch.zeros(proj_dim, dtype=torch.float64, device=self.device),
                "outer": torch.zeros(proj_dim, proj_dim, dtype=torch.float64, device=self.device),
                "reservoir": [],
            }

        was_training = self.model.training
        self.model.eval()

        for batch_idx, batch in enumerate(loader):
            if self.signature_batches > 0 and batch_idx >= self.signature_batches:
                break

            image = batch["img"].to(self.device)
            _, token_layers = self.model.encode_image_with_tokens(
                image, collect_layers=self.signature_layers, use_peer_guidance=False
            )

            for layer in self.signature_layers:
                tokens = token_layers[layer].float().reshape(-1, token_layers[layer].shape[-1])
                tokens = self._deterministic_token_subsample(tokens, self.signature_tokens_per_batch)
                z = tokens @ projection.to(tokens.device)

                z64 = z.double()
                state = accum[layer]
                state["n"] += int(z64.shape[0])
                state["sum"] += z64.sum(dim=0)
                state["outer"] += z64.t().matmul(z64)
                state["reservoir"].append(z.float().cpu())

        if was_training:
            self.model.train()

        layer_stats = {}
        q = torch.tensor(self.signature_quantiles, dtype=torch.float32)

        for layer in self.signature_layers:
            state = accum[layer]
            n = state["n"]
            if n < 2:
                raise RuntimeError(
                    f"Not enough visual tokens to build signature for client {idx}, layer {layer}: n={n}"
                )

            mean = state["sum"] / n
            centered_outer = state["outer"] - n * torch.outer(mean, mean)
            cov = centered_outer / max(n - 1, 1)
            cov = 0.5 * (cov + cov.t())
            cov = cov + 1e-6 * torch.eye(proj_dim, dtype=cov.dtype, device=cov.device)

            reservoir = torch.cat(state["reservoir"], dim=0)
            quantiles = torch.quantile(reservoir, q, dim=0).t().contiguous()

            layer_stats[int(layer)] = {
                "mean": mean.float().cpu(),
                "cov": cov.float().cpu(),
                "quantiles": quantiles.float().cpu(),
                "n_tokens": int(n),
            }

        prompt = self.model.visual_prompt_learner.raw_tokens().float().cpu()
        prompt_delta = self.model.visual_prompt_learner.delta_tokens().float().cpu()

        return {
            "client_id": int(idx),
            "prompt": prompt,
            "prompt_delta": prompt_delta,
            "layers": layer_stats,
            "quantile_levels": tuple(self.signature_quantiles),
            "signature_dim": int(proj_dim),
        }

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            loaded_epoch = checkpoint["epoch"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
