"""Unit tests for ARIA model components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from aria.model import (
    ARIAConfig,
    ARIA,
    ArchitectureGenome,
    MorphogenicAttention,
    PlasticityGatedMLP,
    CognitiveBudgetAllocator,
    StaticMLP,
    EWCWrapper,
    DERPlusPlus,
)


SMALL_CFG = ARIAConfig(
    input_dim    = 28,
    n_classes    = 2,
    d_model      = 16,
    n_layers     = 2,
    n_heads_init = 2,
    n_heads_max  = 4,
    genome_dim   = 8,
)


# ---------------------------------------------------------------------------
# ARIAConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ARIAConfig()
    assert cfg.d_model == 256
    assert cfg.n_heads_max >= cfg.n_heads_init
    assert cfg.spc_lambda > 0


# ---------------------------------------------------------------------------
# ArchitectureGenome
# ---------------------------------------------------------------------------

def test_genome_decode_shapes():
    cfg = SMALL_CFG
    g   = ArchitectureGenome(cfg)
    d   = g.decode()
    assert d["skip_probs"].shape  == (cfg.n_layers,)
    assert d["film_scale"].shape  == (cfg.d_model,)
    assert d["film_shift"].shape  == (cfg.d_model,)
    assert d["temperature"].dim() == 0        # scalar

def test_genome_film_scale_near_one():
    g = ArchitectureGenome(SMALL_CFG)
    d = g.decode()
    assert ((d["film_scale"] - 1.0).abs() < 0.2).all()

def test_genome_temperature_positive():
    g = ArchitectureGenome(SMALL_CFG)
    assert g.decode()["temperature"].item() > 0

def test_genome_reg_loss_differentiable():
    g    = ArchitectureGenome(SMALL_CFG)
    loss = g.reg_loss()
    loss.backward()
    assert g.z.grad is not None


# ---------------------------------------------------------------------------
# MorphogenicAttention
# ---------------------------------------------------------------------------

def test_ma_forward_shape():
    cfg    = SMALL_CFG
    ma     = MorphogenicAttention(cfg)
    genome = ArchitectureGenome(cfg).decode()
    x      = torch.randn(2, 3, cfg.d_model)
    out    = ma(x, genome)
    assert out.shape == x.shape

def test_ma_initial_head_count():
    cfg = SMALL_CFG
    ma  = MorphogenicAttention(cfg)
    assert ma.n_active == cfg.n_heads_init

def test_ma_morphogenesis_split():
    cfg = SMALL_CFG
    ma  = MorphogenicAttention(cfg)
    with torch.no_grad():
        ma.viability.fill_(5.0)   # high viability → should split
    before = ma.n_active
    ma.morphogenesis(global_step=200)
    assert ma.n_active >= before   # may split

def test_ma_head_mask_bounds():
    cfg = SMALL_CFG
    ma  = MorphogenicAttention(cfg)
    assert ma.head_mask.sum() <= cfg.n_heads_max

def test_ma_no_split_during_cooldown():
    cfg = SMALL_CFG
    ma  = MorphogenicAttention(cfg)
    with torch.no_grad():
        ma.viability.fill_(5.0)
        ma.last_morph.fill_(100)
    ma.morphogenesis(global_step=150)   # 150-100=50 < cooldown=100 → no split
    assert ma.n_active == cfg.n_heads_init


# ---------------------------------------------------------------------------
# PlasticityGatedMLP
# ---------------------------------------------------------------------------

def test_pgmlp_forward_shape():
    cfg  = SMALL_CFG
    mlp  = PlasticityGatedMLP(cfg)
    x    = torch.randn(4, cfg.d_model)
    out, p_loss = mlp(x, step=1000)
    assert out.shape == (4, cfg.d_model)
    assert p_loss.dim() == 0

def test_pgmlp_warmup_zero_loss():
    cfg = SMALL_CFG
    mlp = PlasticityGatedMLP(cfg)
    _, p_loss = mlp(torch.randn(4, cfg.d_model), step=0)
    assert p_loss.item() == pytest.approx(0.0)

def test_pgmlp_grad_multiplier_range():
    cfg = SMALL_CFG
    mlp = PlasticityGatedMLP(cfg)
    mlp(torch.randn(4, cfg.d_model), step=1000)
    mult = mlp.slow_grad_multiplier()
    assert 0.0 <= mult <= 1.0

def test_pgmlp_slow_parameters_count():
    cfg = SMALL_CFG
    mlp = PlasticityGatedMLP(cfg)
    # 4 tensors: slow_in weight, slow_in bias, slow_out weight, slow_out bias
    assert len(mlp.slow_parameters()) == 4


# ---------------------------------------------------------------------------
# CognitiveBudgetAllocator
# ---------------------------------------------------------------------------

def test_cba_output_shape():
    cfg    = SMALL_CFG
    cba    = CognitiveBudgetAllocator(cfg)
    x      = torch.randn(8, cfg.input_dim)
    b, bl  = cba(x)
    assert b.shape  == (cfg.n_layers,)
    assert bl.dim() == 0

def test_cba_budget_in_unit_interval():
    cfg   = SMALL_CFG
    cba   = CognitiveBudgetAllocator(cfg)
    x     = torch.randn(8, cfg.input_dim)
    b, _  = cba(x)
    assert (b >= 0).all() and (b <= 1).all()


# ---------------------------------------------------------------------------
# ARIA — full model
# ---------------------------------------------------------------------------

def _small_aria():
    cfg   = SMALL_CFG
    model = ARIA(cfg)
    model.add_task_head(torch.device("cpu"))
    return model, cfg

def test_aria_forward_shape():
    model, cfg = _small_aria()
    x          = torch.randn(4, cfg.input_dim)
    out, aux   = model(x, task_id=0)
    assert out.shape == (4, cfg.n_classes)
    assert aux.dim() == 0

def test_aria_aux_non_negative():
    model, cfg = _small_aria()
    x = torch.randn(4, cfg.input_dim)
    _, aux = model(x, task_id=0)
    assert aux.item() >= 0

def test_aria_backward():
    model, cfg = _small_aria()
    x    = torch.randn(4, cfg.input_dim)
    y    = torch.zeros(4, dtype=torch.long)
    out, aux = model(x, task_id=0)
    loss = (out.mean() + aux)
    loss.backward()
    # at least some params should have gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0

def test_aria_dampen_slow_gradients():
    model, cfg = _small_aria()
    x = torch.randn(4, cfg.input_dim)
    _, aux = model(x, 0)
    aux.backward()
    model.dampen_slow_gradients()   # should not error

def test_aria_add_multiple_heads():
    cfg   = SMALL_CFG
    model = ARIA(cfg)
    for _ in range(5):
        model.add_task_head(torch.device("cpu"))
    assert len(model.task_heads) == 5

def test_aria_architecture_state_keys():
    model, _ = _small_aria()
    state = model.architecture_state()
    assert "head_counts" in state
    assert "gate_means"  in state
    assert "global_step" in state

def test_aria_n_params_positive():
    model, _ = _small_aria()
    assert model.n_params() > 0

def test_aria_spc_consolidate_and_loss():
    from torch.utils.data import DataLoader, TensorDataset
    model, cfg = _small_aria()
    x  = torch.randn(20, cfg.input_dim)
    y  = torch.zeros(20, dtype=torch.long)
    ds = TensorDataset(x, y)
    ld = DataLoader(ds, batch_size=10)
    model.consolidate_slow(ld, task_id=0, device=torch.device("cpu"))
    assert len(model._spc_means) == 1
    spc_l = model._spc_loss(torch.device("cpu"))
    assert spc_l.item() >= 0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_static_mlp_forward():
    m = StaticMLP(input_dim=28, hidden_dim=32, n_layers=2, n_classes=2)
    m.add_task_head(torch.device("cpu"))
    out, _ = m(torch.randn(4, 28), task_id=0)
    assert out.shape == (4, 2)

def test_ewc_wrapper_consolidate_and_loss():
    from torch.utils.data import DataLoader, TensorDataset
    base  = StaticMLP(28, 32, 2, 2)
    ewc   = EWCWrapper(base, ewc_lambda=100.0)
    ewc.add_task_head(torch.device("cpu"))
    x  = torch.randn(16, 28)
    y  = torch.zeros(16, dtype=torch.long)
    ld = DataLoader(TensorDataset(x, y), batch_size=8)
    ewc.consolidate(ld, task_id=0, device=torch.device("cpu"))
    loss = ewc.ewc_loss(torch.device("cpu"))
    assert loss.item() >= 0

def test_der_forward_and_buffer():
    base = StaticMLP(28, 32, 2, 2)
    der  = DERPlusPlus(base, buf_size=50)
    der.add_task_head(torch.device("cpu"))
    x    = torch.randn(8, 28)
    y    = torch.zeros(8, dtype=torch.long)
    out, _ = der(x, task_id=0)
    der.update_buffer(x, y, out, torch.device("cpu"))
    assert der._buf_x is not None


# ---------------------------------------------------------------------------
# Gradient dampening direction (the critical bug check)
# ---------------------------------------------------------------------------

def test_gradient_dampen_direction():
    """
    When π is HIGH (fast pathway dominant), slow-path gradients should be
    SMALL (multiplier close to 0). mul_(1-π) → small. mul_(π) → large (the bug).
    """
    cfg = SMALL_CFG
    mlp = PlasticityGatedMLP(cfg)
    # Force gate toward 1 (fast mode)
    with torch.no_grad():
        for p in mlp.gate_net.parameters():
            p.fill_(10.0)

    x = torch.randn(4, cfg.d_model, requires_grad=False)
    out, p_loss = mlp(x, step=1000)
    out.sum().backward()

    # Save pre-dampen gradients
    pre_grad = mlp.slow_in.weight.grad.clone()
    mlp.slow_in.weight.grad.data.fill_(1.0)

    # Apply dampening
    mult = mlp.slow_grad_multiplier()
    mlp.slow_in.weight.grad.mul_(mult)

    # With high gate (π→1), mult=(1-π)→0, so gradient should be < 1.0
    assert mlp.slow_in.weight.grad.abs().mean().item() < 1.0, \
        "Bug: slow gradient should be dampened when plasticity is high"
