"""
ARIA v2: Adaptive Recurrent Intelligence Architecture
======================================================
Author: Darshan Poudel

Changes from v1/v3
------------------
BUG FIXES:
  [CRITICAL] Gradient dampening inverted: slow pathway now multiplied by (1-π),
             not π. Previously high-plasticity inputs INCREASED slow-path updates —
             the opposite of the intended biology. This explains why PG-MLP ablation
             outperformed ARIA-Full.
  [FIX]      Plasticity loss warmup: inactive for first `warmup_steps` so the
             specialization penalty doesn't dominate task loss at initialization.

NEW CONTRIBUTION — Slow-Pathway Consolidation (SPC):
  EWC-style Fisher regularization applied ONLY to slow-pathway weights.
  Fast pathway remains fully unconstrained → rapid adaptation.
  Slow pathway is Fisher-protected → consolidated retention.
  Key advantage over standard EWC:
    - 50% fewer Fisher params to protect (slow pathway only)
    - More targeted: protects weights the architecture has already identified
      as stable/consolidated (low π) rather than treating all weights equally
    - Synergizes with the gate: on new tasks π naturally rises, routing through
      fast path, leaving slow path protected weights undisturbed

IMPROVEMENTS:
  FiLM conditioning: AGV now applies learned scale+shift (affine transform) to
    attention outputs instead of additive injection. Stronger architectural signal.
  Morphogenesis cooldown: each head tracks last morph event; no split/merge for
    `split_cooldown` steps. Prevents oscillation between split and merge.
  Viability normalization: softmax over active heads so total contribution is
    constant regardless of head count. Stabilizes training after splits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Iterator


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ARIAConfig:
    # Input / task
    input_dim:    int   = 784
    n_classes:    int   = 2

    # Architecture
    d_model:      int   = 256
    n_layers:     int   = 4
    n_heads_init: int   = 4
    n_heads_max:  int   = 8
    genome_dim:   int   = 32
    dropout:      float = 0.1

    # Morphogenesis
    split_threshold: float = 0.65
    merge_threshold: float = 0.97
    split_cooldown:  int   = 100    # global steps between morph events per head
    morph_interval:  int   = 50     # check every N training steps

    # Plasticity
    plasticity_lambda: float = 0.05
    warmup_steps:      int   = 500  # steps before plasticity loss activates

    # Slow-Pathway Consolidation
    spc_lambda: float = 5000.0      # same order as standard EWC lambda

    # Genome regularization
    genome_gamma: float = 0.0001

    # Budget
    budget_beta: float = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Architecture Genome Vector (AGV) — FiLM conditioning
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectureGenome(nn.Module):
    """
    Differentiable latent vector encoding structural hyperparameters.
    Co-optimized with weights. FiLM (scale+shift) provides stronger
    architectural influence than additive conditioning.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        G, D, L = cfg.genome_dim, cfg.d_model, cfg.n_layers
        self.z = nn.Parameter(torch.randn(G) * 0.01)
        self.proj_skip  = nn.Linear(G, L)
        self.proj_temp  = nn.Linear(G, 1)
        # FiLM: near-identity init (scale ≈ 1, shift ≈ 0)
        self.proj_scale = nn.Linear(G, D)
        self.proj_shift = nn.Linear(G, D)

    def decode(self) -> dict:
        z = self.z
        return {
            "skip_probs":  torch.sigmoid(self.proj_skip(z)),          # (L,)
            "temperature": F.softplus(self.proj_temp(z)).squeeze() + 0.5,
            "film_scale":  1.0 + 0.1 * torch.tanh(self.proj_scale(z)),  # (D,) near 1
            "film_shift":  0.1  * torch.tanh(self.proj_shift(z)),        # (D,) near 0
        }

    def reg_loss(self) -> torch.Tensor:
        return 0.5 * (self.z ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Morphogenic Attention (MA)
# ─────────────────────────────────────────────────────────────────────────────

class MorphogenicAttention(nn.Module):
    """
    Pre-allocated H_max parameter tensors; a boolean mask activates heads.
    Heads split (specialize) or merge (consolidate) based on viability.

    Viability interpretation:
      High v_i → head carries heavy load → split to distribute work.
      High cos-sim(W_Q_i, W_Q_j) → heads redundant → merge.

    Softmax-normalized viabilities: total contribution is constant across
    different head counts, stabilizing optimization post-morphogenesis.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        H   = cfg.n_heads_max
        D   = cfg.d_model
        d_h = D // H
        self.d_h     = d_h
        self.D       = D
        self.split_τ = cfg.split_threshold
        self.merge_τ = cfg.merge_threshold
        self.cooldown = cfg.split_cooldown

        self.W_Q = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_K = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_V = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_O = nn.Parameter(torch.randn(H, d_h, D) * 0.02)
        self.out_proj = nn.Linear(D, D, bias=False)

        # Learned viability per slot; negative init so heads earn activation
        self.viability = nn.Parameter(torch.zeros(H) - 0.5)

        mask = torch.zeros(H, dtype=torch.bool)
        mask[:cfg.n_heads_init] = True
        self.register_buffer("head_mask", mask)

        # Last global step each slot was involved in a morph event
        self.register_buffer("last_morph", torch.zeros(H, dtype=torch.long))

        self.dropout = nn.Dropout(cfg.dropout)

    @property
    def n_active(self) -> int:
        return int(self.head_mask.sum().item())

    def forward(self, x: torch.Tensor, genome: dict) -> torch.Tensor:
        B, T, _ = x.shape
        τ = genome["temperature"].clamp(min=0.1)

        active = self.head_mask.nonzero(as_tuple=True)[0]

        # Normalize viabilities so total output magnitude is stable
        v_raw = torch.stack([self.viability[i] for i in active])
        v_norm = torch.softmax(v_raw, dim=0) * len(active)  # sums to n_active

        outputs = []
        for k, i in enumerate(active):
            Q = x @ self.W_Q[i]   # (B, T, d_h)
            K = x @ self.W_K[i]
            V = x @ self.W_V[i]
            scores = (Q @ K.transpose(-2, -1)) / (math.sqrt(self.d_h) * τ)
            if T > 1:
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal, float('-inf'))
            attn = self.dropout(F.softmax(scores, dim=-1))
            out  = (attn @ V) @ self.W_O[i]   # (B, T, D)
            outputs.append(v_norm[k] * out)

        result = torch.stack(outputs, 0).sum(0)   # (B, T, D)
        result = self.out_proj(result)

        # FiLM modulation from genome
        scale = genome["film_scale"].to(x.device)
        shift = genome["film_shift"].to(x.device)
        result = scale * result + shift            # broadcast over B, T

        return result

    def morphogenesis(self, global_step: int):
        """
        Split overloaded heads; merge redundant heads.
        Both operations respect per-head cooldown.
        """
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        H_max  = self.head_mask.shape[0]
        newly_involved: set = set()

        # ── Split ─────────────────────────────────────────────────────────────
        for i in active:
            if self.n_active >= H_max:
                break
            if global_step - int(self.last_morph[i].item()) < self.cooldown:
                continue
            v = torch.sigmoid(self.viability[i]).item()
            if v <= self.split_τ:
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
            newly_involved.add(i)
            newly_involved.add(j)

        # ── Merge ─────────────────────────────────────────────────────────────
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        merged: set = set()
        for k in range(len(active) - 1):
            i, j = active[k], active[k + 1]
            if i in merged or j in merged:
                continue
            if i in newly_involved or j in newly_involved:
                continue
            if global_step - int(self.last_morph[i].item()) < self.cooldown:
                continue
            if global_step - int(self.last_morph[j].item()) < self.cooldown:
                continue
            if self.n_active <= 2:
                break
            cos = F.cosine_similarity(
                self.W_Q[i].flatten().unsqueeze(0),
                self.W_Q[j].flatten().unsqueeze(0)
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


# ─────────────────────────────────────────────────────────────────────────────
# Plasticity-Gated MLP (PG-MLP)
# ─────────────────────────────────────────────────────────────────────────────

class PlasticityGatedMLP(nn.Module):
    """
    Dual-pathway MLP: fast (high π, volatile) and slow (low π, stable).

    FIXED gradient dampening: slow pathway now multiplied by (1-π).
    When π is high (fast mode), (1-π) is small → slow weights barely updated.
    When π is low (slow mode), (1-π) is large → slow weights update normally.
    The previous code used π, which was the exact inverse.

    ADDED warmup: plasticity specialization loss is gated off for the first
    `warmup_steps` steps so it doesn't swamp the task loss at initialization.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        D, d_ff = cfg.d_model, cfg.d_model * 2
        self.fast_in  = nn.Linear(D, d_ff)
        self.fast_out = nn.Linear(d_ff, D)
        self.slow_in  = nn.Linear(D, d_ff)
        self.slow_out = nn.Linear(d_ff, D)
        self.gate_net = nn.Sequential(
            nn.Linear(D, d_ff // 4),
            nn.ReLU(),
            nn.Linear(d_ff // 4, 1),
            nn.Sigmoid()
        )
        self.dropout       = nn.Dropout(cfg.dropout)
        self.lambda_       = cfg.plasticity_lambda
        self.warmup_steps  = cfg.warmup_steps
        self.mean_gate     = 0.5   # tracked externally for gradient dampening

    def forward(self, x: torch.Tensor, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        π = self.gate_net(x)                         # (B, T, 1)
        self.mean_gate = float(π.detach().mean().item())

        h_fast = F.gelu(self.fast_in(x))
        h_slow = F.gelu(self.slow_in(x))
        out = π * self.fast_out(h_fast) + (1 - π) * self.slow_out(h_slow)
        out = self.dropout(out)

        if global_step >= self.warmup_steps:
            p_loss = self.lambda_ / (π * (1 - π) + 1e-4).mean()
        else:
            p_loss = torch.zeros(1, device=x.device).squeeze()

        return out, p_loss

    def slow_parameters(self) -> List[nn.Parameter]:
        """Return slow-pathway parameters only (used for SPC and gradient dampening)."""
        return (
            list(self.slow_in.parameters()) +
            list(self.slow_out.parameters())
        )

    def slow_grad_multiplier(self) -> float:
        """(1 - π̄): small when plasticity is high, protects slow path."""
        return 1.0 - self.mean_gate


# ─────────────────────────────────────────────────────────────────────────────
# Cognitive Budget Allocator (CBA)
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveBudgetAllocator(nn.Module):
    """
    Predicts per-layer compute budget b_l ∈ [0,1] from raw input statistics.
    Uses raw input (not hidden state) so signal differs meaningfully across tasks.
    """
    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.n_layers),
            nn.Sigmoid()
        )
        self.beta = cfg.budget_beta
        self.input_dim = cfg.input_dim

    def forward(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = x_raw.device
        f1 = x_raw.std(dim=-1).mean().unsqueeze(0)
        p  = F.softmax(x_raw.abs(), dim=-1)
        f2 = -(p * (p + 1e-8).log()).sum(dim=-1).mean().unsqueeze(0) / math.log(self.input_dim)
        f3 = (x_raw.max(dim=-1).values - x_raw.min(dim=-1).values).mean().unsqueeze(0)
        budgets = self.net(torch.stack([f1, f2, f3]).squeeze())
        return budgets, self.beta * budgets.mean()


# ─────────────────────────────────────────────────────────────────────────────
# ARIA Block
# ─────────────────────────────────────────────────────────────────────────────

class ARIABlock(nn.Module):
    def __init__(self, cfg: ARIAConfig, idx: int):
        super().__init__()
        self.idx = idx
        D = cfg.d_model
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = MorphogenicAttention(cfg)
        self.mlp  = PlasticityGatedMLP(cfg)

    def forward(self, x: torch.Tensor, genome: dict,
                budget: torch.Tensor, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.attn(self.ln1(x), genome) + x
        h, pl = self.mlp(self.ln2(z), global_step)
        b     = budget[self.idx]
        return b * (z + h) + (1 - b) * x, pl


# ─────────────────────────────────────────────────────────────────────────────
# Full ARIA Model
# ─────────────────────────────────────────────────────────────────────────────

class ARIA(nn.Module):
    """
    ARIA v2 with Slow-Pathway Consolidation (SPC).

    SPC principle:
      Standard EWC applies Fisher regularization to ALL weights equally.
      But ARIA's PG-MLP already differentiates weights:
        - Slow pathway weights: low π, stable, consolidated
        - Fast pathway weights: high π, volatile, adaptive
      SPC exploits this by regularizing ONLY slow-pathway weights.
      Result: fast pathway adapts freely (no penalty), slow pathway
      retains prior knowledge (Fisher-protected). This is more targeted
      than EWC and requires half the Fisher memory.

    Usage:
      After training on task t, call model.consolidate_slow(loader, t, device).
      SPC loss is automatically added in forward() for subsequent tasks.
    """

    def __init__(self, cfg: ARIAConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        self.input_proj = nn.Linear(cfg.input_dim, D)
        self.genome     = ArchitectureGenome(cfg)
        self.blocks     = nn.ModuleList([ARIABlock(cfg, i) for i in range(cfg.n_layers)])
        self.cba        = CognitiveBudgetAllocator(cfg)
        self.ln_f       = nn.LayerNorm(D)
        self.task_heads = nn.ModuleList()

        # SPC state: one entry per completed task
        self._spc_means:   List[Dict[str, torch.Tensor]] = []
        self._spc_fishers: List[Dict[str, torch.Tensor]] = []

        self.global_step  = 0
        self.n_tasks_seen = 0

    # ── Task management ───────────────────────────────────────────────────────

    def add_task_head(self, device: torch.device, n_classes: int = None):
        nc   = n_classes if n_classes is not None else self.cfg.n_classes
        head = nn.Linear(self.cfg.d_model, nc).to(device)
        self.task_heads.append(head)
        self.n_tasks_seen += 1

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        budgets, b_loss = self.cba(x)
        h = F.gelu(self.input_proj(x)).unsqueeze(1)   # (B, 1, D)
        genome = self.genome.decode()

        total_p = torch.zeros(1, device=device).squeeze()
        for block in self.blocks:
            if self.training:
                skip_p = genome["skip_probs"][block.idx].item() * 0.1
                if torch.rand(1).item() < skip_p:
                    continue
            h, p = block(h, genome, budgets, self.global_step)
            total_p = total_p + p

        h   = self.ln_f(h).squeeze(1)                 # (B, D)
        out = self.task_heads[task_id](h)

        if self.training:
            self.global_step += 1
            if self.global_step % self.cfg.morph_interval == 0:
                for block in self.blocks:
                    block.attn.morphogenesis(self.global_step)

        aux = (total_p + b_loss
               + self.cfg.genome_gamma * self.genome.reg_loss()
               + self._spc_loss(device))
        return out, aux

    # ── Slow-Pathway Consolidation ────────────────────────────────────────────

    def _slow_named_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for i, block in enumerate(self.blocks):
            mlp = block.mlp
            for attr, param in [
                (f"b{i}.slow_in.w",  mlp.slow_in.weight),
                (f"b{i}.slow_in.b",  mlp.slow_in.bias),
                (f"b{i}.slow_out.w", mlp.slow_out.weight),
                (f"b{i}.slow_out.b", mlp.slow_out.bias),
            ]:
                yield attr, param

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

    def consolidate_slow(self, loader, task_id: int, device: torch.device):
        """
        Compute diagonal Fisher for slow-pathway weights after task `task_id`.
        Call once after training on each task (before moving to the next).
        """
        self.eval()
        means   = {n: p.detach().cpu().clone() for n, p in self._slow_named_params()}
        fishers = {n: torch.zeros_like(p).cpu() for n, p in self._slow_named_params()}
        n = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out, _ = self(x, task_id)
            log_p  = F.log_softmax(out, dim=1)
            log_p[range(len(y)), y].sum().backward()
            for name, param in self._slow_named_params():
                if param.grad is not None:
                    fishers[name] += param.grad.data.cpu() ** 2
            n += 1

        for name in fishers:
            fishers[name] /= max(n, 1)

        self._spc_means.append(means)
        self._spc_fishers.append(fishers)
        self.train()

    # ── Gradient dampening ────────────────────────────────────────────────────

    def dampen_slow_gradients(self):
        """
        Called after loss.backward(), before optimizer.step().
        Multiplies slow-pathway gradients by (1 - π̄).
        FIXED: was mul_(π), now correctly mul_(1 - π).
        """
        for block in self.blocks:
            mult = block.mlp.slow_grad_multiplier()   # 1 - mean_gate
            for p in block.mlp.slow_parameters():
                if p.grad is not None:
                    p.grad.mul_(mult)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def architecture_state(self) -> dict:
        return {
            "head_counts": [b.attn.n_active for b in self.blocks],
            "total_heads": sum(b.attn.n_active for b in self.blocks),
            "gate_means":  [round(b.mlp.mean_gate, 4) for b in self.blocks],
            "global_step": self.global_step,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (parameter-scalable)
# ─────────────────────────────────────────────────────────────────────────────

class StaticMLP(nn.Module):
    """
    Flat 4-layer MLP baseline. Scale hidden_dim to match ARIA parameter count.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                       nn.Dropout(dropout)]
        self.body       = nn.Sequential(*layers)
        self.task_heads = nn.ModuleList()
        self.n_tasks_seen = 0
        self._hidden_dim = hidden_dim

    def add_task_head(self, device: torch.device, n_classes: int = 2):
        self.task_heads.append(nn.Linear(self._hidden_dim, n_classes).to(device))
        self.n_tasks_seen += 1

    def forward(self, x: torch.Tensor,
                task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h   = self.body(x)
        out = self.task_heads[task_id](h)
        return out, torch.zeros(1, device=x.device).squeeze()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EWCWrapper(nn.Module):
    """
    Standard EWC wrapping a StaticMLP. Fisher computed over ALL parameters.
    """
    def __init__(self, base: StaticMLP, ewc_lambda: float = 5000.0):
        super().__init__()
        self.model      = base
        self.ewc_lambda = ewc_lambda
        self._means:   List[Dict[str, torch.Tensor]] = []
        self._fishers: List[Dict[str, torch.Tensor]] = []

    def add_task_head(self, device: torch.device, n_classes: int = 2):
        self.model.add_task_head(device, n_classes)

    @property
    def n_tasks_seen(self): return self.model.n_tasks_seen

    def forward(self, x: torch.Tensor,
                task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.model(x, task_id)
        return out, torch.zeros(1, device=x.device).squeeze()

    def ewc_loss(self, device: torch.device) -> torch.Tensor:
        if not self._means:
            return torch.zeros(1, device=device).squeeze()
        loss = torch.zeros(1, device=device).squeeze()
        for means, fishers in zip(self._means, self._fishers):
            for name, param in self.model.named_parameters():
                if name in means:
                    loss = loss + (
                        fishers[name].to(device) *
                        (param - means[name].to(device)) ** 2
                    ).sum()
        return self.ewc_lambda * loss

    def consolidate(self, loader, task_id: int, device: torch.device):
        self.eval()
        means   = {n: p.detach().cpu().clone()
                   for n, p in self.model.named_parameters()}
        fishers = {n: torch.zeros_like(p).cpu()
                   for n, p in self.model.named_parameters()}
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out, _ = self.model(x, task_id)
            log_p  = F.log_softmax(out, dim=1)
            log_p[range(len(y)), y].sum().backward()
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
    """
    Dark Experience Replay ++ baseline.
    Stores a replay buffer of (x, logits) pairs from past tasks.
    Regularizes with α * MSE(current logits, stored logits) for old samples.
    Adds β * CE(current logits, stored labels) for task boundary correction.
    """
    def __init__(self, base: StaticMLP, buffer_size: int = 200,
                 alpha: float = 0.1, beta: float = 0.5):
        super().__init__()
        self.model       = base
        self.buffer_size = buffer_size
        self.alpha       = alpha
        self.beta        = beta
        self._buf_x:      Optional[torch.Tensor] = None
        self._buf_logits: Optional[torch.Tensor] = None
        self._buf_y:      Optional[torch.Tensor] = None

    def add_task_head(self, device: torch.device, n_classes: int = 2):
        self.model.add_task_head(device, n_classes)

    @property
    def n_tasks_seen(self): return self.model.n_tasks_seen

    def forward(self, x: torch.Tensor,
                task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.model(x, task_id)
        return out, torch.zeros(1, device=x.device).squeeze()

    def der_loss(self, device: torch.device, task_id: int) -> torch.Tensor:
        if self._buf_x is None:
            return torch.zeros(1, device=device).squeeze()
        bx = self._buf_x.to(device)
        bl = self._buf_logits.to(device)
        by = self._buf_y.to(device)
        cur_logits, _ = self.model(bx, task_id)
        # Match logit distribution (DER)
        der_term = self.alpha * F.mse_loss(cur_logits, bl)
        # Correct labels (DER++)
        derpp_term = self.beta * F.cross_entropy(cur_logits, by)
        return der_term + derpp_term

    def update_buffer(self, x: torch.Tensor, y: torch.Tensor,
                      logits: torch.Tensor, device: torch.device):
        x      = x.detach().cpu()
        y      = y.detach().cpu()
        logits = logits.detach().cpu()
        if self._buf_x is None:
            self._buf_x      = x[:self.buffer_size]
            self._buf_logits = logits[:self.buffer_size]
            self._buf_y      = y[:self.buffer_size]
        else:
            n   = min(len(x), self.buffer_size)
            idx = torch.randperm(self.buffer_size + n)[:self.buffer_size]
            bx  = torch.cat([self._buf_x, x[:n]], 0)[idx]
            bl  = torch.cat([self._buf_logits, logits[:n]], 0)[idx]
            by  = torch.cat([self._buf_y, y[:n]], 0)[idx]
            self._buf_x      = bx
            self._buf_logits = bl
            self._buf_y      = by

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = ARIAConfig(input_dim=784, d_model=256, n_layers=4)
    model = ARIA(cfg)
    x     = torch.randn(4, 784)
    model.add_task_head(torch.device("cpu"))
    out, loss = model(x, 0)
    print(f"Output shape : {out.shape}")
    print(f"Aux loss     : {loss.item():.4f}")
    print(f"Params       : {model.n_params():,}")
    print(f"Head counts  : {model.architecture_state()['head_counts']}")
    print("Sanity check passed.")
