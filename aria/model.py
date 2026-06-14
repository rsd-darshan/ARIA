"""
ARIA — Adaptive Recurrent Intelligence Architecture
Model definitions.

Classes
-------
ARIAConfig     : dataclass holding all hyperparameters
ARIA           : full model (MA + PG-MLP + AGV + CBA + SPC)
StaticMLP      : fixed-width MLP baseline
EWCWrapper     : EWC wrapping StaticMLP
DERPlusPlus    : Dark Experience Replay ++ baseline
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ARIAConfig:
    input_dim:         int   = 784
    n_classes:         int   = 2
    d_model:           int   = 256
    n_layers:          int   = 4
    n_heads_init:      int   = 4
    n_heads_max:       int   = 8
    genome_dim:        int   = 32
    dropout:           float = 0.1
    split_threshold:   float = 0.65
    merge_threshold:   float = 0.97
    split_cooldown:    int   = 100
    morph_interval:    int   = 50
    plasticity_lambda: float = 0.05
    warmup_steps:      int   = 500
    spc_lambda:        float = 5000.0
    genome_gamma:      float = 0.0001
    budget_beta:       float = 0.001
    # ARIA-v2 additions
    spad_lambda:       float = 50.0   # SPAD: L2 anchor on slow pathway weights
    slow_lr_ratio:     float = 0.5    # asymmetric LR: slow params get lr * slow_lr_ratio
    # ARIA-Final additions
    adapter_dim:       int   = 64     # task-specific fast adapter bottleneck dimension


# ---------------------------------------------------------------------------
# Architecture Genome Vector — FiLM conditioning
# ---------------------------------------------------------------------------

class ArchitectureGenome(nn.Module):
    """
    Differentiable latent vector z ∈ R^G encoding structural hyperparameters.
    Co-optimized with weights via backprop.

    Decodes into:
      skip_probs  : (L,)  per-layer skip probability
      temperature : scalar  attention temperature > 0.5
      film_scale  : (D,)  near 1.0 — affine scale for layer outputs
      film_shift  : (D,)  near 0.0 — affine shift for layer outputs

    FiLM conditioning provides stronger architectural signal than additive injection.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        G, D, L = cfg.genome_dim, cfg.d_model, cfg.n_layers
        self.z          = nn.Parameter(torch.randn(G) * 0.01)
        self.proj_skip  = nn.Linear(G, L)
        self.proj_temp  = nn.Linear(G, 1)
        self.proj_scale = nn.Linear(G, D)
        self.proj_shift = nn.Linear(G, D)

    def decode(self) -> Dict[str, torch.Tensor]:
        z = self.z
        return {
            "skip_probs":  torch.sigmoid(self.proj_skip(z)),
            "temperature": F.softplus(self.proj_temp(z)).squeeze() + 0.5,
            "film_scale":  1.0 + 0.1 * torch.tanh(self.proj_scale(z)),
            "film_shift":  0.1 * torch.tanh(self.proj_shift(z)),
        }

    def reg_loss(self) -> torch.Tensor:
        return 0.5 * (self.z ** 2).mean()


# ---------------------------------------------------------------------------
# Morphogenic Attention
# ---------------------------------------------------------------------------

class MorphogenicAttention(nn.Module):
    """
    Multi-head attention with dynamic head count.

    Mechanism
    ---------
    Heads are pre-allocated up to H_max; a boolean mask activates them.
    Every `morph_interval` steps:
      Split : heads with viability > split_threshold spawn a child (perturbed copy).
      Merge : adjacent heads with cosine similarity > merge_threshold are averaged.

    A per-head cooldown prevents oscillation immediately after a morph event.

    Viabilities are softmax-normalised so total output magnitude stays constant
    across different active head counts.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        H, D = cfg.n_heads_max, cfg.d_model
        d_h  = D // H
        self.d_h      = d_h
        self.split_τ  = cfg.split_threshold
        self.merge_τ  = cfg.merge_threshold
        self.cooldown = cfg.split_cooldown

        self.W_Q       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_K       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_V       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_O       = nn.Parameter(torch.randn(H, d_h, D) * 0.02)
        self.out_proj  = nn.Linear(D, D, bias=False)
        self.viability = nn.Parameter(torch.zeros(H) - 0.5)

        mask = torch.zeros(H, dtype=torch.bool)
        mask[:cfg.n_heads_init] = True
        self.register_buffer("head_mask",  mask)
        self.register_buffer("last_morph", torch.zeros(H, dtype=torch.long))
        self.dropout = nn.Dropout(cfg.dropout)

    @property
    def n_active(self) -> int:
        return int(self.head_mask.sum().item())

    def forward(self, x: torch.Tensor, genome: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, T, _ = x.shape
        τ       = genome["temperature"].clamp(min=0.1)
        active  = self.head_mask.nonzero(as_tuple=True)[0]

        v_raw  = torch.stack([self.viability[i] for i in active])
        v_norm = torch.softmax(v_raw, dim=0) * len(active)

        outputs = []
        for k, i in enumerate(active):
            Q = x @ self.W_Q[i]
            K = x @ self.W_K[i]
            V = x @ self.W_V[i]
            s = (Q @ K.transpose(-2, -1)) / (math.sqrt(self.d_h) * τ)
            if T > 1:
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                s = s.masked_fill(~causal, float("-inf"))
            attn = self.dropout(F.softmax(s, dim=-1))
            outputs.append(v_norm[k] * (attn @ V) @ self.W_O[i])

        result = torch.stack(outputs, 0).sum(0)
        result = self.out_proj(result)
        result = genome["film_scale"] * result + genome["film_shift"]
        return result

    def morphogenesis(self, global_step: int) -> None:
        active  = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        H_max   = self.head_mask.shape[0]
        touched: set = set()

        # Split
        for i in active:
            if self.n_active >= H_max:
                break
            if global_step - int(self.last_morph[i].item()) < self.cooldown:
                continue
            if torch.sigmoid(self.viability[i]).item() <= self.split_τ:
                continue
            inactive = (~self.head_mask).nonzero(as_tuple=True)[0]
            if len(inactive) == 0:
                break
            j = inactive[0].item()
            with torch.no_grad():
                for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                    W[j] = W[i] + 0.01 * torch.randn_like(W[i])
                self.viability[j] = self.viability[i].clone() - 0.5
                self.last_morph[i] = global_step
                self.last_morph[j] = global_step
            self.head_mask[j] = True
            touched.update([i, j])

        # Merge
        active  = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        merged: set = set()
        for k in range(len(active) - 1):
            i, j = active[k], active[k + 1]
            if i in merged or j in merged or i in touched or j in touched:
                continue
            if global_step - int(self.last_morph[i].item()) < self.cooldown:
                continue
            if global_step - int(self.last_morph[j].item()) < self.cooldown:
                continue
            if self.n_active <= 2:
                break
            cos = F.cosine_similarity(
                self.W_Q[i].flatten().unsqueeze(0),
                self.W_Q[j].flatten().unsqueeze(0),
            ).item()
            if cos > self.merge_τ:
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[i] = (W[i] + W[j]) / 2
                    self.viability[i] = torch.max(self.viability[i], self.viability[j])
                    self.last_morph[i] = global_step
                merged.add(j)
        for j in merged:
            self.head_mask[j] = False


# ---------------------------------------------------------------------------
# Plasticity-Gated MLP
# ---------------------------------------------------------------------------

class PlasticityGatedMLP(nn.Module):
    """
    Dual fast/slow pathway MLP.

    Gate π ∈ (0,1) per token routes computation:
      output = π · fast(x) + (1−π) · slow(x)

    Biological analogy: fast pathway ≈ hippocampus (rapid, volatile);
    slow pathway ≈ neocortex (stable, consolidated).

    Specialisation loss pushes π toward 0 or 1 (bimodal), activated only
    after warmup_steps to prevent early optimisation conflict.

    Gradient dampening: slow-pathway gradients are multiplied by (1−π̄),
    so fast/volatile inputs update the slow path minimally.
    NOTE: earlier versions used π (inverted); this is the corrected version.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        D, d_ff = cfg.d_model, cfg.d_model * 2
        self.fast_in   = nn.Linear(D, d_ff)
        self.fast_out  = nn.Linear(d_ff, D)
        self.slow_in   = nn.Linear(D, d_ff)
        self.slow_out  = nn.Linear(d_ff, D)
        self.gate_net  = nn.Sequential(
            nn.Linear(D, d_ff // 4), nn.ReLU(),
            nn.Linear(d_ff // 4, 1), nn.Sigmoid(),
        )
        self.dropout      = nn.Dropout(cfg.dropout)
        self.lambda_      = cfg.plasticity_lambda
        self.warmup_steps = cfg.warmup_steps
        self.mean_gate    = 0.5

    def forward(
        self, x: torch.Tensor, step: int, force_slow: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if force_slow:
            # task-conditioned gate: old tasks route entirely through consolidated slow pathway
            π = torch.zeros(*x.shape[:-1], 1, device=x.device)
        else:
            π = self.gate_net(x)
        self.mean_gate = float(π.detach().mean().item())

        h_fast = F.gelu(self.fast_in(x))
        h_slow = F.gelu(self.slow_in(x))
        out    = π * self.fast_out(h_fast) + (1 - π) * self.slow_out(h_slow)
        out    = self.dropout(out)

        if step >= self.warmup_steps:
            p_loss = self.lambda_ / (π * (1 - π) + 1e-4).mean()
        else:
            p_loss = torch.zeros(1, device=x.device).squeeze()

        return out, p_loss

    def slow_parameters(self) -> List[nn.Parameter]:
        return list(self.slow_in.parameters()) + list(self.slow_out.parameters())

    def slow_grad_multiplier(self) -> float:
        """(1 − π̄): close to 0 when plasticity is high → protect slow path."""
        return 1.0 - self.mean_gate


# ---------------------------------------------------------------------------
# Cognitive Budget Allocator
# ---------------------------------------------------------------------------

class CognitiveBudgetAllocator(nn.Module):
    """
    Predicts per-layer compute budget b_l ∈ [0,1] from raw input statistics.
    Uses pixel-level features (std, entropy proxy, range) so signal is
    task-discriminative and not hidden-state uniform.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, cfg.n_layers), nn.Sigmoid(),
        )
        self.beta      = cfg.budget_beta
        self.input_dim = cfg.input_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = x.std(dim=-1).mean().unsqueeze(0)
        p  = F.softmax(x.abs(), dim=-1)
        f2 = -(p * (p + 1e-8).log()).sum(dim=-1).mean().unsqueeze(0) / math.log(self.input_dim)
        f3 = (x.max(dim=-1).values - x.min(dim=-1).values).mean().unsqueeze(0)
        b  = self.net(torch.stack([f1, f2, f3]).squeeze())
        return b, self.beta * b.mean()


# ---------------------------------------------------------------------------
# Task-Specific Fast Adapter (ARIA-Final)
# ---------------------------------------------------------------------------

class TaskFastAdapter(nn.Module):
    """
    Per-task bottleneck residual adapter applied to the final representation.
    Starts as identity (up weights zero-initialised), learns task-specific
    fast-pathway features. Frozen after task training ends — new tasks never
    overwrite old task adapters, eliminating a key source of forgetting.
    """
    def __init__(self, d_model: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim)
        self.up   = nn.Linear(adapter_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.relu(self.down(x)))


# ---------------------------------------------------------------------------
# ARIA Block
# ---------------------------------------------------------------------------

class ARIABlock(nn.Module):
    def __init__(self, cfg: ARIAConfig, idx: int):
        super().__init__()
        D        = cfg.d_model
        self.idx = idx
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = MorphogenicAttention(cfg)
        self.mlp  = PlasticityGatedMLP(cfg)

    def forward(
        self, x: torch.Tensor, genome: Dict, budget: torch.Tensor, step: int,
        force_slow: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.attn(self.ln1(x), genome) + x
        h, pl = self.mlp(self.ln2(z), step, force_slow=force_slow)
        b     = budget[self.idx]
        return b * (z + h) + (1 - b) * x, pl


# ---------------------------------------------------------------------------
# Full ARIA Model
# ---------------------------------------------------------------------------

class ARIA(nn.Module):
    """
    Adaptive Recurrent Intelligence Architecture.

    Four core mechanisms
    --------------------
    1. Morphogenic Attention (MA): head count adapts via split / merge.
    2. Plasticity-Gated MLP (PG-MLP): fast/slow dual pathway with learned gate.
    3. Architecture Genome Vector (AGV): shared latent z conditions all blocks.
    4. Cognitive Budget Allocator (CBA): per-layer compute budget from input complexity.

    Fifth mechanism (post-training, per task)
    -----------------------------------------
    5. Slow-Pathway Consolidation (SPC): Fisher-based EWC applied only to slow-pathway
       weights. Fast pathway remains unconstrained. Call consolidate_slow() after each task.

    Parameters
    ----------
    cfg : ARIAConfig

    Usage
    -----
    model = ARIA(cfg)
    model.add_task_head(device)
    out, aux_loss = model(x, task_id=0)
    # after task 0:
    model.consolidate_slow(train_loader, task_id=0, device=device)
    """

    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        self.cfg        = cfg
        D               = cfg.d_model
        self.input_proj = nn.Linear(cfg.input_dim, D)
        self.genome     = ArchitectureGenome(cfg)
        self.blocks     = nn.ModuleList([ARIABlock(cfg, i) for i in range(cfg.n_layers)])
        self.cba        = CognitiveBudgetAllocator(cfg)
        self.ln_f       = nn.LayerNorm(D)
        self.task_heads:    nn.ModuleList = nn.ModuleList()
        self.fast_adapters: nn.ModuleList = nn.ModuleList()

        self._spc_means:   List[Dict[str, torch.Tensor]] = []
        self._spc_fishers: List[Dict[str, torch.Tensor]] = []
        self._slow_snapshot: Dict[str, torch.Tensor] = {}  # SPAD: frozen post-consolidation slow weights

        self.global_step    = 0
        self.n_tasks_seen   = 0
        self.current_task_id = 0  # updated by train loop; used for task-conditioned gate routing

    # ------------------------------------------------------------------
    # Task heads
    # ------------------------------------------------------------------

    def add_task_head(self, device: torch.device, n_classes: Optional[int] = None) -> None:
        nc = n_classes if n_classes is not None else self.cfg.n_classes
        self.task_heads.append(nn.Linear(self.cfg.d_model, nc).to(device))
        self.fast_adapters.append(
            TaskFastAdapter(self.cfg.d_model, self.cfg.adapter_dim).to(device)
        )
        self.n_tasks_seen += 1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, task_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device          = x.device
        budgets, b_loss = self.cba(x)
        h               = F.gelu(self.input_proj(x)).unsqueeze(1)
        genome          = self.genome.decode()
        total_p         = torch.zeros(1, device=device).squeeze()

        # task-conditioned gate: at eval time, route old tasks through slow pathway only
        force_slow = (not self.training) and (task_id < self.current_task_id)

        for block in self.blocks:
            if self.training:
                if torch.rand(1).item() < genome["skip_probs"][block.idx].item() * 0.1:
                    continue
            h, p    = block(h, genome, budgets, self.global_step, force_slow=force_slow)
            total_p = total_p + p

        h   = self.ln_f(h).squeeze(1)
        h   = self.fast_adapters[task_id](h)  # task-specific adapter (frozen for old tasks)
        out = self.task_heads[task_id](h)

        if self.training:
            self.global_step += 1
            if self.global_step % self.cfg.morph_interval == 0:
                for block in self.blocks:
                    block.attn.morphogenesis(self.global_step)

        aux = (total_p + b_loss
               + self.cfg.genome_gamma * self.genome.reg_loss()
               + self._spc_loss(device)
               + self._spad_loss(device))
        return out, aux

    # ------------------------------------------------------------------
    # Slow-Pathway Consolidation (SPC)
    # ------------------------------------------------------------------

    def _slow_named_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for i, block in enumerate(self.blocks):
            m = block.mlp
            for key, p in [
                (f"b{i}.slow_in.w",  m.slow_in.weight),
                (f"b{i}.slow_in.b",  m.slow_in.bias),
                (f"b{i}.slow_out.w", m.slow_out.weight),
                (f"b{i}.slow_out.b", m.slow_out.bias),
            ]:
                yield key, p

    def _spc_loss(self, device: torch.device) -> torch.Tensor:
        if not self._spc_means:
            return torch.zeros(1, device=device).squeeze()
        loss = torch.zeros(1, device=device).squeeze()
        for means, fishers in zip(self._spc_means, self._spc_fishers):
            for name, param in self._slow_named_params():
                if name in means:
                    loss = loss + (
                        fishers[name].to(device) *
                        (param - means[name].to(device)) ** 2
                    ).sum()
        return self.cfg.spc_lambda * loss

    def consolidate_slow(
        self,
        loader: torch.utils.data.DataLoader,
        task_id: int,
        device: torch.device,
    ) -> None:
        """
        Compute diagonal Fisher information for slow-pathway weights after task_id.
        Must be called after each task's training, before the next task starts.
        """
        self.eval()
        means   = {n: p.detach().cpu().clone() for n, p in self._slow_named_params()}
        fishers = {n: torch.zeros_like(p).cpu() for n, p in self._slow_named_params()}
        n = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out, _ = self(x, task_id)
            F.log_softmax(out, dim=1)[range(len(y)), y].sum().backward()
            for name, param in self._slow_named_params():
                if param.grad is not None:
                    fishers[name] += param.grad.data.cpu() ** 2
            n += 1

        for name in fishers:
            fishers[name] /= max(n, 1)

        self._spc_means.append(means)
        self._spc_fishers.append(fishers)
        self.snapshot_slow()  # SPAD: freeze post-consolidation slow weights
        self.train()

    # ------------------------------------------------------------------
    # SPAD — Slow-Pathway Activation Distillation (ARIA-v2)
    # ------------------------------------------------------------------

    def slow_parameters(self) -> List[nn.Parameter]:
        """All slow-pathway parameters across all blocks, for asymmetric LR."""
        params: List[nn.Parameter] = []
        for block in self.blocks:
            params.extend(block.mlp.slow_parameters())
        return params

    def freeze_task_adapter(self, task_id: int) -> None:
        """Freeze task adapter after training — new tasks can never overwrite it."""
        for p in self.fast_adapters[task_id].parameters():
            p.requires_grad_(False)

    def snapshot_slow(self) -> None:
        """Store frozen copy of current slow weights. Called after each consolidation."""
        self._slow_snapshot = {
            n: p.detach().cpu().clone() for n, p in self._slow_named_params()
        }

    def _spad_loss(self, device: torch.device) -> torch.Tensor:
        """L2 distance from current slow weights to their post-consolidation snapshot.
        Prevents slow pathway from drifting while learning new tasks."""
        if not self._slow_snapshot:
            return torch.zeros(1, device=device).squeeze()
        loss = torch.zeros(1, device=device).squeeze()
        for name, param in self._slow_named_params():
            if name in self._slow_snapshot:
                loss = loss + ((param - self._slow_snapshot[name].to(device)) ** 2).sum()
        return self.cfg.spad_lambda * loss

    # ------------------------------------------------------------------
    # Gradient dampening — must call after loss.backward()
    # ------------------------------------------------------------------

    def dampen_slow_gradients(self) -> None:
        """
        Multiply slow-pathway gradients by (1 − π̄).
        High plasticity → small multiplier → slow path barely updated.
        """
        for block in self.blocks:
            mult = block.mlp.slow_grad_multiplier()
            for p in block.mlp.slow_parameters():
                if p.grad is not None:
                    p.grad.mul_(mult)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def architecture_state(self) -> Dict:
        return {
            "head_counts": [b.attn.n_active for b in self.blocks],
            "total_heads": sum(b.attn.n_active for b in self.blocks),
            "gate_means":  [round(b.mlp.mean_gate, 4) for b in self.blocks],
            "global_step": self.global_step,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class StaticMLP(nn.Module):
    """Fixed-width 4-layer MLP baseline. Scale hidden_dim to match ARIA params."""

    def __init__(
        self,
        input_dim:  int   = 784,
        hidden_dim: int   = 256,
        n_layers:   int   = 4,
        n_classes:  int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        layers: list = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        self.body       = nn.Sequential(*layers)
        self.task_heads: nn.ModuleList = nn.ModuleList()
        self._hidden    = hidden_dim
        self.n_classes  = n_classes

    def add_task_head(self, device: torch.device, n_classes: Optional[int] = None) -> None:
        nc = n_classes if n_classes is not None else self.n_classes
        self.task_heads.append(nn.Linear(self._hidden, nc).to(device))

    def forward(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.task_heads[task_id](self.body(x)), torch.zeros(1, device=x.device).squeeze()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EWCWrapper(nn.Module):
    """EWC over all parameters of a StaticMLP."""

    def __init__(self, base: StaticMLP, ewc_lambda: float = 5000.0):
        super().__init__()
        self.model      = base
        self.ewc_lambda = ewc_lambda
        self._means:   list = []
        self._fishers: list = []

    def add_task_head(self, device: torch.device, n_classes: Optional[int] = None) -> None:
        self.model.add_task_head(device, n_classes)

    @property
    def n_tasks_seen(self) -> int:
        return len(self.model.task_heads)

    def forward(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, task_id)

    def ewc_loss(self, device: torch.device) -> torch.Tensor:
        if not self._means:
            return torch.zeros(1, device=device).squeeze()
        loss = torch.zeros(1, device=device).squeeze()
        for means, fishers in zip(self._means, self._fishers):
            for name, param in self.model.named_parameters():
                if name in means:
                    loss = loss + (fishers[name].to(device) * (param - means[name].to(device)) ** 2).sum()
        return self.ewc_lambda * loss

    def consolidate(self, loader, task_id: int, device: torch.device) -> None:
        self.eval()
        means   = {n: p.detach().cpu().clone()  for n, p in self.model.named_parameters()}
        fishers = {n: torch.zeros_like(p).cpu() for n, p in self.model.named_parameters()}
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out, _ = self.model(x, task_id)
            F.log_softmax(out, dim=1)[range(len(y)), y].sum().backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fishers[name] += param.grad.data.cpu() ** 2
            n += 1
        for name in fishers:
            fishers[name] /= max(n, 1)
        self._means.append(means)
        self._fishers.append(fishers)
        self.train()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DERPlusPlus(nn.Module):
    """Dark Experience Replay ++."""

    def __init__(self, base: StaticMLP, buf_size: int = 200, alpha: float = 0.1, beta: float = 0.5):
        super().__init__()
        self.model    = base
        self.buf_size = buf_size
        self.alpha    = alpha
        self.beta     = beta
        self._buf_x:      Optional[torch.Tensor] = None
        self._buf_logits: Optional[torch.Tensor] = None
        self._buf_y:      Optional[torch.Tensor] = None

    def add_task_head(self, device: torch.device, n_classes: Optional[int] = None) -> None:
        self.model.add_task_head(device, n_classes)

    @property
    def n_tasks_seen(self) -> int:
        return len(self.model.task_heads)

    def forward(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, task_id)

    def der_loss(self, device: torch.device, task_id: int) -> torch.Tensor:
        if self._buf_x is None:
            return torch.zeros(1, device=device).squeeze()
        bx = self._buf_x.to(device)
        bl = self._buf_logits.to(device)
        by = self._buf_y.to(device)
        cur, _ = self.model(bx, task_id)
        return self.alpha * F.mse_loss(cur, bl) + self.beta * F.cross_entropy(cur, by)

    def update_buffer(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, _device) -> None:
        x, y, logits = x.detach().cpu(), y.detach().cpu(), logits.detach().cpu()
        if self._buf_x is None:
            self._buf_x      = x[:self.buf_size]
            self._buf_logits = logits[:self.buf_size]
            self._buf_y      = y[:self.buf_size]
        else:
            n          = min(len(x), self.buf_size)
            combined_x = torch.cat([self._buf_x,      x[:n]], 0)
            combined_l = torch.cat([self._buf_logits, logits[:n]], 0)
            combined_y = torch.cat([self._buf_y,      y[:n]], 0)
            keep       = min(self.buf_size, len(combined_x))
            idx        = torch.randperm(len(combined_x))[:keep]
            self._buf_x      = combined_x[idx]
            self._buf_logits = combined_l[idx]
            self._buf_y      = combined_y[idx]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
