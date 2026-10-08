"""PORT_PLAN Step 1: LoRA for the PyTorch pi0.5, matching openpi's JAX LoRA.

Run on a GPU machine (needs the pi05_base PyTorch weights and the golden file for the GPU tests):

    OPENPI_G1_WEIGHTS=<dir with model.safetensors> OPENPI_G1_GOLDEN=<golden .pt> \
        uv run pytest tests/g1/test_lora.py -v
"""

import dataclasses
import os
import pathlib
import sys

# JAX is only used for abstract shapes here; keep it off the GPU (it would pre-allocate ~75% of the memory).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models_pytorch import lora as _lora
from openpi.models_pytorch import pi0_pytorch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "g1"))
import make_golden as mg  # noqa: E402

WEIGHTS = os.environ.get("OPENPI_G1_WEIGHTS")
GOLDEN = os.environ.get("OPENPI_G1_GOLDEN")
needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and WEIGHTS and GOLDEN), reason="needs CUDA, OPENPI_G1_WEIGHTS, OPENPI_G1_GOLDEN"
)

LORA_CFG = mg.model_config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")


# ---------------------------------------------------------------- math (CPU)


def _jax_style_out(x, w_a, w_b):
    """JAX q/k/v einsum LoRA: w_a (N, D, r), w_b (N, r, H): `BTD,NDr->BTNr`, `BTNr,NrH->BTNH`."""
    return torch.einsum("btnr,nrh->btnh", torch.einsum("btd,ndr->btnr", x, w_a), w_b)


def _jax_style_in(x, w_a, w_b):
    """JAX attn_vec einsum LoRA: x (B, T, N, H), w_a (N, H, r), w_b (N, r, D): `BTNH,NHr->BTNr`, `BTNr,NrD->BTD`."""
    return torch.einsum("btnr,nrd->btd", torch.einsum("btnh,nhr->btnr", x, w_a), w_b)


@pytest.mark.parametrize("heads", [1, 4])
def test_headwise_out_matches_jax_einsum(heads):
    torch.manual_seed(0)
    n, d, h, r = heads, 8, 3, 2
    base = torch.nn.Linear(d, n * h, bias=False)
    lin = _lora.LoRALinear(base, rank=r, scaling=0.5, heads=n, head_side="out")
    x = torch.randn(2, 5, d)
    w_a = lin.lora_a.permute(0, 2, 1)  # (N, r, D) -> JAX (N, D, r)
    w_b = lin.lora_b.permute(0, 2, 1)  # (N, H, r) -> JAX (N, r, H)
    expected = base(x) + 0.5 * _jax_style_out(x, w_a, w_b).reshape(2, 5, n * h)
    torch.testing.assert_close(lin(x), expected)
    torch.testing.assert_close(torch.nn.functional.linear(x, lin.merged_weight()), expected)


def test_headwise_in_matches_jax_einsum():
    torch.manual_seed(0)
    n, h, d, r = 4, 3, 8, 2
    base = torch.nn.Linear(n * h, d, bias=False)
    lin = _lora.LoRALinear(base, rank=r, scaling=1.0, heads=n, head_side="in")
    x = torch.randn(2, 5, n * h)
    w_a = lin.lora_a.permute(0, 2, 1)  # (N, r, H) -> JAX (N, H, r)
    w_b = lin.lora_b.permute(0, 2, 1)  # (N, D, r) -> JAX (N, r, D)
    expected = base(x) + _jax_style_in(x.reshape(2, 5, n, h), w_a, w_b)
    torch.testing.assert_close(lin(x), expected)
    torch.testing.assert_close(torch.nn.functional.linear(x, lin.merged_weight()), expected)


def test_lora_keeps_base_parameter_names():
    base = torch.nn.Linear(8, 6)
    lin = _lora.LoRALinear(base, rank=2, scaling=1.0, heads=2)
    assert set(lin.state_dict()) == {"weight", "bias", "lora_a", "lora_b"}
    assert lin.weight is base.weight


# ---------------------------------------------------------------- structure vs JAX (CPU, abstract)


def _jax_counts(config: pi0_config.Pi0Config) -> dict[str, int]:
    """LoRA and trainable parameter counts of the JAX model for the same config, under its freeze filter."""
    from flax import nnx

    model = nnx.eval_shape(config.create, jax.random.key(0))
    state = nnx.state(model, nnx.Param)
    frozen = nnx.state(model, config.get_freeze_filter())
    total = sum(int(np.prod(x.value.shape)) for x in jax.tree.leaves(state, is_leaf=lambda v: hasattr(v, "value")))
    frozen_n = sum(int(np.prod(x.value.shape)) for x in jax.tree.leaves(frozen, is_leaf=lambda v: hasattr(v, "value")))
    lora = nnx.state(model, nnx.All(nnx.Param, nnx_utils_lora_filter()))
    lora_n = sum(int(np.prod(x.value.shape)) for x in jax.tree.leaves(lora, is_leaf=lambda v: hasattr(v, "value")))
    return {"total": total, "trainable": total - frozen_n, "lora": lora_n}


def nnx_utils_lora_filter():
    from openpi.shared import nnx_utils

    return nnx_utils.PathRegex(".*lora.*")


def _torch_counts(config: pi0_config.Pi0Config) -> dict[str, int]:
    with torch.device("meta"):
        model = pi0_pytorch.PI0Pytorch(dataclasses.replace(config, dtype="float32"))
    _lora.freeze_like_jax(model, config.paligemma_variant, config.action_expert_variant)
    return _lora.count_params(model)


@pytest.mark.parametrize(
    "variants",
    [("gemma_2b_lora", "gemma_300m_lora"), ("gemma_2b", "gemma_300m_lora"), ("gemma_2b_lora", "gemma_300m")],
)
def test_lora_and_trainable_counts_match_jax(variants):
    config = mg.model_config(paligemma_variant=variants[0], action_expert_variant=variants[1])
    jax_counts, torch_counts = _jax_counts(config), _torch_counts(config)
    print(f"{variants}: JAX {jax_counts}  torch {torch_counts}")
    assert torch_counts["lora"] == jax_counts["lora"]
    assert torch_counts["trainable"] == jax_counts["trainable"]


# ---------------------------------------------------------------- real weights (GPU)


@pytest.fixture(scope="module")
def lora_model():
    with torch.device("cuda"):
        model = pi0_pytorch.PI0Pytorch(LORA_CFG)
    missing, unexpected = safetensors.torch.load_model(
        model, os.path.join(WEIGHTS, "model.safetensors"), strict=False, device="cuda"
    )
    _lora.check_pretrained_load(missing, unexpected)
    assert missing and all(_lora.is_lora_param(k) for k in missing)
    mg.strict_numerics()
    return model.eval()


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN)


def _lora_modules(model):
    return [m for m in model.modules() if isinstance(m, _lora.LoRALinear)]


@needs_gpu
def test_zero_b_reproduces_golden(lora_model, golden):
    """With every B = 0 the LoRA path adds exactly 0: outputs equal the base model's golden outputs."""
    saved = [m.lora_b.detach().clone() for m in _lora_modules(lora_model)]
    with torch.no_grad():
        for m in _lora_modules(lora_model):
            m.lora_b.zero_()
    try:
        outputs = mg.run_reference(lora_model, golden["inputs"], "cuda")
        assert mg.compare(golden["outputs"], outputs)
    finally:
        with torch.no_grad():
            for m, b in zip(_lora_modules(lora_model), saved, strict=True):
                m.lora_b.copy_(b)


@needs_gpu
def test_one_step_updates_only_trainable(lora_model, golden):
    """After freeze_like_jax, a training step produces gradients exactly on SigLIP, projector, projections and LoRA."""
    _lora.freeze_like_jax(lora_model, LORA_CFG.paligemma_variant, LORA_CFG.action_expert_variant)
    lora_model.train()
    try:
        trainable = {n for n, p in lora_model.named_parameters() if p.requires_grad}
        assert any(_lora.is_lora_param(n) for n in trainable)
        assert any("vision_tower" in n for n in trainable)
        assert not any(".language_model." in n and not _lora.is_lora_param(n) for n in trainable)
        assert not any("gemma_expert" in n and not _lora.is_lora_param(n) for n in trainable)

        observation = mg.to_observation(golden["inputs"], "cuda")
        actions, noise, time = (golden["inputs"][k].cuda() for k in ("actions", "noise", "time"))
        lora_model.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        loss = lora_model(observation, actions, noise=noise, time=time).mean()
        loss.backward()
        # Only the action tokens' output reaches the loss. The VLM's last layer still feeds the action tokens through
        # its keys/values, but its own o_proj -> MLP output is unused, so those LoRA params get no gradient (as in JAX).
        last = f".language_model.layers.{len(lora_model.paligemma_with_expert.paligemma.language_model.layers) - 1}."
        expected_no_grad = {
            n for n in trainable if last in n and _lora.is_lora_param(n) and (".o_proj." in n or ".mlp." in n)
        }
        no_grad = {n for n in trainable if dict(lora_model.named_parameters())[n].grad is None}
        assert no_grad == expected_no_grad, sorted(no_grad ^ expected_no_grad)[:10]
        assert all(p.grad is None for p in lora_model.parameters() if not p.requires_grad), "grad on a frozen param"
        a_before = _lora_modules(lora_model)[0].lora_a.detach().clone()
        torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=1e-4).step()
        assert not torch.equal(a_before, _lora_modules(lora_model)[0].lora_a)
    finally:
        lora_model.zero_grad(set_to_none=True)
        for p in lora_model.parameters():
            p.requires_grad_(True)
        lora_model.eval()


@needs_gpu
def test_merge_equals_unmerged(lora_model, golden):
    """Folding B @ A into the weights: exact per layer (fp32 rounding), and negligible end to end. Runs last (merges in place)."""
    with torch.no_grad():
        for m in _lora_modules(lora_model):
            m.lora_b.mul_(20.0)  # make the LoRA effect clearly visible
        for m in _lora_modules(lora_model):
            x = torch.randn(3, m.in_features, device="cuda")
            y = m(x)
            y_merged = torch.nn.functional.linear(x, m.merged_weight(), m.bias)
            assert ((y - y_merged).abs().max() / y.abs().max()).item() < 1e-5
    unmerged = mg.run_reference(lora_model, golden["inputs"], "cuda")
    _lora.merge_lora(lora_model)
    assert not _lora_modules(lora_model)
    merged = mg.run_reference(lora_model, golden["inputs"], "cuda")
    for key in unmerged:
        effect = (unmerged[key] - golden["outputs"][key]).abs().max().item()
        error = (merged[key] - unmerged[key]).abs().max().item()
        print(f"{key}: LoRA effect {effect:.3e}, merge error {error:.3e}")
        assert effect > 1e-2, "LoRA should change the output"
        assert error < 1e-3 * effect


def test_trainable_to_float32_keeps_frozen_dtype():
    """JAX recipe: trainable params fp32, frozen params stay in the training precision (bf16)."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(torch.bfloat16)
    model[0].requires_grad_(False)
    assert _lora.trainable_to_float32(model) == 2
    assert all(p.dtype == torch.bfloat16 for p in model[0].parameters())
    assert all(p.dtype == torch.float32 for p in model[1].parameters())
