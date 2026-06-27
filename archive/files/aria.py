"""
ARIA: Adaptive Recurrent Intelligence Architecture
A Self-Evolving Neural Framework for General Artificial Intelligence

Author: Darshan Poudel
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ARIAConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 6
    n_heads_init: int = 8          # Starting number of attention heads
    n_heads_max: int = 16          # Max heads after splitting
    d_ff_init: int = 2048          # Starting MLP hidden dim
    genome_dim: int = 64           # AGV dimension
    dropout: float = 0.1
    split_threshold: float = 0.7   # s_i > τ → trigger head split
    merge_threshold: float = 0.92  # cos sim > τ → trigger head merge
    split_noise_scale: float = 0.01
    plasticity_lambda: float = 0.01
    budget_beta: float = 0.001
    genome_gamma: float = 0.0001
    max_heads_per_layer: int = 16


# ──────────────────────────────────────────────────────────────────────────────
# Architecture Genome Vector (AGV)
# ──────────────────────────────────────────────────────────────────────────────

class ArchitectureGenome(nn.Module):
    """
    Global latent vector encoding structural hyperparameters.
    Co-optimized with model weights during training.
    """
    def __init__(self, config: ARIAConfig):
        super().__init__()
        G = config.genome_dim
        self.z = nn.Parameter(torch.randn(G) * 0.01)

        # Projection heads decode structural hyperparameters
        self.proj_depth     = nn.Linear(G, config.n_layers)   # layer skip probs
        self.proj_temp      = nn.Linear(G, 1)                 # attention temperature
        self.proj_expansion = nn.Linear(G, 1)                 # MLP expansion ratio
        self.proj_cond      = nn.Linear(G, config.d_model)    # conditioning signal

    def decode(self) -> dict:
        z = self.z
        skip_probs  = torch.sigmoid(self.proj_depth(z))       # (L,)
        temperature = F.softplus(self.proj_temp(z)) + 0.5     # > 0.5
        expansion   = 2.0 + 2.0 * torch.sigmoid(self.proj_expansion(z))  # (2,4)
        cond_signal = torch.tanh(self.proj_cond(z))           # (D,)
        return {
            "skip_probs":  skip_probs,
            "temperature": temperature.squeeze(),
            "expansion":   expansion.squeeze(),
            "cond_signal": cond_signal,
        }

    def regularization_loss(self) -> torch.Tensor:
        return 0.5 * (self.z ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Morphogenic Attention (MA)
# ──────────────────────────────────────────────────────────────────────────────

class MorphogenicAttention(nn.Module):
    """
    Multi-head attention where heads have learnable viability scores
    and can split (specialize) or merge (consolidate) during training.
    """
    def __init__(self, config: ARIAConfig):
        super().__init__()
        self.d_model    = config.d_model
        self.n_heads    = config.n_heads_init
        self.max_heads  = config.max_heads_per_layer
        self.split_τ    = config.split_threshold
        self.merge_τ    = config.merge_threshold
        self.noise_σ    = config.split_noise_scale

        d_h = config.d_model // config.n_heads_init
        self.d_h = d_h

        # Per-head projections (stored as list for dynamic size changes)
        self._init_heads(config.n_heads_init, config.d_model, d_h)

        # Viability and split score learners
        self.viability_net  = nn.Linear(d_h, 1)   # v_i from head output
        self.split_net      = nn.Linear(d_h * d_h, 1)  # s_i from weight variance
        self.output_proj    = nn.Linear(config.d_model, config.d_model)
        self.dropout        = nn.Dropout(config.dropout)

        # Genome conditioning
        self.genome_proj = nn.Linear(config.genome_dim, config.d_model)

    def _init_heads(self, n_heads: int, d_model: int, d_h: int):
        self.W_Q = nn.ParameterList([nn.Parameter(torch.randn(d_model, d_h) * 0.02)
                                     for _ in range(n_heads)])
        self.W_K = nn.ParameterList([nn.Parameter(torch.randn(d_model, d_h) * 0.02)
                                     for _ in range(n_heads)])
        self.W_V = nn.ParameterList([nn.Parameter(torch.randn(d_model, d_h) * 0.02)
                                     for _ in range(n_heads)])
        self.W_O = nn.ParameterList([nn.Parameter(torch.randn(d_h, d_model) * 0.02)
                                     for _ in range(n_heads)])
        self.n_heads = n_heads

    def _head_attention(self, x: torch.Tensor, i: int,
                        temperature: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention for head i. Returns (output, attn_weights)."""
        B, T, _ = x.shape
        Q = x @ self.W_Q[i]   # (B, T, d_h)
        K = x @ self.W_K[i]
        V = x @ self.W_V[i]

        scale = (self.d_h ** 0.5) * temperature
        scores = (Q @ K.transpose(-2, -1)) / scale  # (B, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = attn @ V          # (B, T, d_h)
        return out, attn

    def _compute_viability(self, head_out: torch.Tensor) -> torch.Tensor:
        """Compute viability score for a head output."""
        pooled = head_out.mean(dim=(0, 1))  # (d_h,)
        return torch.sigmoid(self.viability_net(pooled.unsqueeze(0))).squeeze()

    def _compute_split_score(self, i: int) -> torch.Tensor:
        """Compute split readiness: high variance in weights → good to split."""
        W = self.W_Q[i]  # (d_model, d_h)
        var_vec = W.var(dim=0)  # variance across input dims per output dim
        return torch.sigmoid(self.split_net(var_vec.flatten().unsqueeze(0))).squeeze()

    def maybe_split_head(self, i: int):
        """
        If head i's split score > threshold, spawn two child heads.
        Uses soft perturbation to maintain gradient continuity.
        """
        if self.n_heads >= self.max_heads:
            return False

        split_score = self._compute_split_score(i)
        if split_score.item() > self.split_τ:
            noise = self.noise_σ
            for W_list in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                W_orig = W_list[i].data.clone()
                W_list[i] = nn.Parameter(W_orig + noise * torch.randn_like(W_orig))
                W_list.append(nn.Parameter(W_orig - noise * torch.randn_like(W_orig)))
            self.n_heads += 1
            return True
        return False

    def maybe_merge_heads(self, i: int, j: int) -> bool:
        """If heads i and j are too similar, merge them."""
        if i >= self.n_heads or j >= self.n_heads or i == j:
            return False

        cos_sim = F.cosine_similarity(
            self.W_Q[i].flatten().unsqueeze(0),
            self.W_Q[j].flatten().unsqueeze(0)
        ).item()

        if cos_sim > self.merge_τ:
            for W_list in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                merged = nn.Parameter((W_list[i].data + W_list[j].data) / 2)
                W_list[i] = merged
                del W_list[j]
            self.n_heads -= 1
            return True
        return False

    def forward(self, x: torch.Tensor, genome: dict,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        temperature = genome["temperature"]

        head_outputs = []
        viabilities = []

        for i in range(self.n_heads):
            h_out, _ = self._head_attention(x, i, temperature, mask)
            v_i       = self._compute_viability(h_out)
            contrib   = h_out @ self.W_O[i]    # (B, T, D)
            head_outputs.append(contrib * v_i)
            viabilities.append(v_i)

        # Aggregate weighted head outputs
        out = torch.stack(head_outputs, dim=0).sum(dim=0)  # (B, T, D)

        # Genome conditioning
        cond = self.genome_proj(genome["cond_signal"])  # (D,)
        out  = out + cond.unsqueeze(0).unsqueeze(0)

        return self.output_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# Plasticity-Gated MLP (PG-MLP)
# ──────────────────────────────────────────────────────────────────────────────

class PlasticityGatedMLP(nn.Module):
    """
    Dual-pathway MLP with per-neuron plasticity gates.
    Fast pathway (high π): learns quickly, prone to forgetting.
    Slow pathway (low π): stable, consolidates knowledge.
    """
    def __init__(self, config: ARIAConfig):
        super().__init__()
        d_model = config.d_model
        d_ff    = config.d_ff_init

        # Fast pathway
        self.W_fast_in  = nn.Linear(d_model, d_ff)
        self.W_fast_out = nn.Linear(d_ff, d_model)

        # Slow pathway (smaller, more stable)
        self.W_slow_in  = nn.Linear(d_model, d_ff)
        self.W_slow_out = nn.Linear(d_ff, d_model)

        # Plasticity gate (context-dependent)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_ff // 4),
            nn.ReLU(),
            nn.Linear(d_ff // 4, d_ff),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(config.dropout)
        self.lambda_ = config.plasticity_lambda

        # Register hooks to dampen slow pathway gradients
        self._mean_gate = 0.5  # updated dynamically

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # π: plasticity gate  (B, T, d_ff)
        π = self.gate_net(x)
        self._mean_gate = π.mean().item()

        # Fast pathway
        h_fast = F.gelu(self.W_fast_in(x))     # (B, T, d_ff)
        out_fast = self.W_fast_out(h_fast)       # (B, T, D)

        # Slow pathway
        h_slow = F.gelu(self.W_slow_in(x))
        out_slow = self.W_slow_out(h_slow)

        # Merge: π gates fast, (1-π) gates slow  (broadcast over B, T)
        π_out = π.mean(dim=-1, keepdim=True)    # scalar gate per position
        out   = π_out * out_fast + (1 - π_out) * out_slow
        out   = self.dropout(out)

        # Plasticity loss: push gates toward 0 or 1 (specialization)
        plasticity_loss = self.lambda_ * (1.0 / (π * (1 - π) + 1e-6)).mean()

        return out, plasticity_loss

    def get_slow_weight_multiplier(self) -> float:
        """Return dampening factor for slow pathway weight updates."""
        return self._mean_gate  # high plasticity → slow weights updated less


# ──────────────────────────────────────────────────────────────────────────────
# Cognitive Budget Allocator (CBA)
# ──────────────────────────────────────────────────────────────────────────────

class CognitiveBudgetAllocator(nn.Module):
    """
    Lightweight meta-network that predicts per-layer compute budget
    from input complexity signals. Output b_l ∈ [0,1] per layer.
    """
    def __init__(self, config: ARIAConfig):
        super().__init__()
        n_layers  = config.n_layers
        input_dim = 2  # (entropy, residual norm) → simple complexity signal

        self.budget_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_layers),
            nn.Sigmoid()
        )
        self.beta = config.budget_beta

    def compute_complexity(self, x: torch.Tensor,
                           prev_residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute a simple complexity vector from current activations."""
        # Entropy proxy: std of activations (high std → uncertain → complex)
        entropy_proxy = x.std(dim=-1).mean().unsqueeze(0)

        # Residual norm: how much is changing
        if prev_residual is not None:
            res_norm = (x - prev_residual).norm(dim=-1).mean().unsqueeze(0)
        else:
            res_norm = x.norm(dim=-1).mean().unsqueeze(0)

        # Normalize to (0,1)
        complexity = torch.stack([
            torch.sigmoid(entropy_proxy),
            torch.sigmoid(res_norm)
        ])
        return complexity  # (2,)

    def forward(self, x: torch.Tensor,
                prev_x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        complexity = self.compute_complexity(x, prev_x)
        budgets    = self.budget_net(complexity)   # (n_layers,)
        budget_loss = self.beta * budgets.mean()   # sparsity penalty
        return budgets, budget_loss


# ──────────────────────────────────────────────────────────────────────────────
# ARIA Block
# ──────────────────────────────────────────────────────────────────────────────

class ARIABlock(nn.Module):
    """Single ARIA block: MA → PG-MLP → Budget gate."""
    def __init__(self, config: ARIAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = MorphogenicAttention(config)
        self.mlp  = PlasticityGatedMLP(config)

    def forward(self, x: torch.Tensor, genome: dict, budget: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Morphogenic Attention
        attn_out = self.attn(self.ln1(x), genome, mask)
        x = x + attn_out

        # PG-MLP
        mlp_out, plasticity_loss = self.mlp(self.ln2(x))

        # Cognitive Budget gating: interpolate between residual and full update
        b = budget[self.layer_idx].unsqueeze(-1).unsqueeze(-1)  # broadcast
        x = b * (x + mlp_out) + (1 - b) * x

        return x, plasticity_loss


# ──────────────────────────────────────────────────────────────────────────────
# Full ARIA Model
# ──────────────────────────────────────────────────────────────────────────────

class ARIA(nn.Module):
    """
    Full ARIA model for language modeling (easily adaptable to other tasks).
    
    Key properties:
    - Architecture evolves during training (head splitting/merging)
    - Fast/slow memory pathways per layer
    - Global genome vector conditions all blocks
    - Adaptive compute via per-layer budget
    """
    def __init__(self, config: ARIAConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop      = nn.Dropout(config.dropout)

        self.genome    = ArchitectureGenome(config)
        self.blocks    = nn.ModuleList([ARIABlock(config, i) for i in range(config.n_layers)])
        self.budget_allocator = CognitiveBudgetAllocator(config)

        self.ln_f  = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Morphogenesis step counter
        self.register_buffer('morph_step', torch.tensor(0))
        self.morph_interval = 500  # check for splits/merges every N steps

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(T, T, device=device))

    def morphogenesis_step(self):
        """
        Attempt head splits and merges across all blocks.
        Called periodically during training.
        """
        for block in self.blocks:
            attn = block.attn
            n = attn.n_heads

            # Try splitting each head
            for i in range(n):
                attn.maybe_split_head(i)

            # Try merging adjacent heads
            n = attn.n_heads
            for i in range(0, n - 1, 2):
                attn.maybe_merge_heads(i, i + 1)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok = self.token_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=device))
        x   = self.drop(tok + pos)

        # Decode genome
        genome = self.genome.decode()

        # Compute budgets from input
        budgets, budget_loss = self.budget_allocator(x)

        # Causal mask
        mask = self._causal_mask(T, device)

        # Forward through blocks
        total_plasticity_loss = torch.tensor(0.0, device=device)
        skip_probs = genome["skip_probs"]

        prev_x = x.clone()
        for i, block in enumerate(self.blocks):
            # Stochastic layer skip based on genome skip probability
            if self.training and torch.rand(1).item() < skip_probs[i].item() * 0.2:
                continue  # Skip this layer (scaled down to avoid too much skipping)

            x, p_loss = block(x, genome, budgets, mask)
            total_plasticity_loss = total_plasticity_loss + p_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # Compute total loss
        loss = None
        if targets is not None:
            task_loss     = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            genome_reg    = self.config.genome_gamma * self.genome.regularization_loss()
            loss          = task_loss + total_plasticity_loss + budget_loss + genome_reg

        # Morphogenesis check
        if self.training:
            self.morph_step += 1
            if self.morph_step.item() % self.morph_interval == 0:
                self.morphogenesis_step()

        return logits, loss

    def get_architecture_state(self) -> dict:
        """Return current architecture statistics."""
        head_counts = [block.attn.n_heads for block in self.blocks]
        genome_decoded = self.genome.decode()
        gates = []
        for block in self.blocks:
            gates.append(block.mlp._mean_gate)

        return {
            "head_counts":       head_counts,
            "total_heads":       sum(head_counts),
            "genome_temperature": genome_decoded["temperature"].item(),
            "genome_expansion":  genome_decoded["expansion"].item(),
            "layer_skip_probs":  genome_decoded["skip_probs"].detach().cpu().tolist(),
            "mean_plasticity_gates": gates,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ARIATrainer:
    """Simple training wrapper for ARIA."""

    def __init__(self, model: ARIA, lr: float = 3e-4, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000)

    def train_step(self, input_ids: torch.Tensor,
                   targets: torch.Tensor) -> dict:
        self.model.train()
        input_ids = input_ids.to(self.device)
        targets   = targets.to(self.device)

        self.optimizer.zero_grad()
        logits, loss = self.model(input_ids, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Dampen slow pathway gradients
        for block in self.model.blocks:
            mult = block.mlp.get_slow_weight_multiplier()
            for param in block.mlp.W_slow_in.parameters():
                if param.grad is not None:
                    param.grad *= mult
            for param in block.mlp.W_slow_out.parameters():
                if param.grad is not None:
                    param.grad *= mult

        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "arch": self.model.get_architecture_state()
        }

    @torch.no_grad()
    def evaluate(self, input_ids: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.eval()
        input_ids = input_ids.to(self.device)
        targets   = targets.to(self.device)
        _, loss   = self.model(input_ids, targets)
        return loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Quick Sanity Check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ARIA: Adaptive Recurrent Intelligence Architecture")
    print("=" * 60)

    config = ARIAConfig(
        vocab_size=1000,
        max_seq_len=64,
        d_model=128,
        n_layers=4,
        n_heads_init=4,
        d_ff_init=256,
        genome_dim=32,
    )

    model   = ARIA(config)
    trainer = ARIATrainer(model, lr=1e-3, device="cpu")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Fake batch
    B, T = 4, 32
    ids     = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    print("Initial architecture state:")
    arch = model.get_architecture_state()
    for k, v in arch.items():
        print(f"  {k}: {v}")
    print()

    # Run a few training steps
    print("Training steps:")
    for step in range(5):
        result = trainer.train_step(ids, targets)
        print(f"  Step {step+1} | Loss: {result['loss']:.4f} | "
              f"Heads: {result['arch']['head_counts']} | "
              f"Temp: {result['arch']['genome_temperature']:.3f}")

    print()
    print("Post-training architecture state:")
    arch = model.get_architecture_state()
    for k, v in arch.items():
        print(f"  {k}: {v}")

    print()
    print("ARIA initialized and running successfully!")
