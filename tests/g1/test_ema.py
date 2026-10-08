"""PORT_PLAN Step 1b: EMA in the PyTorch trainer, matching the JAX trainer. CPU only.

    uv run pytest tests/g1/test_ema.py -v
"""

import dataclasses
import importlib.util
import pathlib
import types

import safetensors.torch
import torch

from openpi.training import ema_pytorch as _ema

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _trainer():
    spec = importlib.util.spec_from_file_location("train_pytorch", ROOT / "scripts" / "train_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    model[0].weight.requires_grad_(False)  # a frozen param
    return model


def _step(model, opt, ema):
    loss = model(torch.randn(16, 4)).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema.update()


def test_update_matches_jax_formula():
    model = _toy()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expected = {n: p.detach().clone() for n, p in trainable.items()}  # JAX: ema starts at the initial params
    ema = _ema.ParamEMA(model, decay=0.9)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    for _ in range(5):
        _step(model, opt, ema)
        for n, p in trainable.items():
            expected[n] = 0.9 * expected[n] + 0.1 * p.detach()
    for n, e in zip(ema.names, ema.shadow, strict=True):
        torch.testing.assert_close(e, expected[n])


def test_frozen_params_are_not_tracked():
    model = _toy()
    ema = _ema.ParamEMA(model, decay=0.99)
    assert "0.weight" not in ema.names
    assert set(ema.names) == {"0.bias", "2.weight", "2.bias"}


def test_swapped_restores_raw_exactly():
    model = _toy()
    ema = _ema.ParamEMA(model, decay=0.5)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    _step(model, opt, ema)
    raw = {n: p.detach().clone() for n, p in model.named_parameters()}
    with ema.swapped():
        for n, e in zip(ema.names, ema.shadow, strict=True):
            assert torch.equal(dict(model.named_parameters())[n], e)
    for n, p in model.named_parameters():
        assert torch.equal(p, raw[n])


@dataclasses.dataclass
class _Cfg:
    checkpoint_dir: pathlib.Path
    save_interval: int = 1
    num_train_steps: int = 100
    wandb_enabled: bool = False


def test_checkpoint_saves_ema_for_inference_and_resumes(tmp_path):
    trainer = _trainer()
    data_config = types.SimpleNamespace(norm_stats=None, asset_id=None)
    cfg = _Cfg(checkpoint_dir=tmp_path)

    model = _toy()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ema = _ema.ParamEMA(model, decay=0.8)
    for _ in range(3):
        _step(model, opt, ema)
    trainer.save_checkpoint(model, opt, 3, cfg, True, data_config, ema=ema)

    # model.safetensors = EMA weights (what serving loads), raw weights in raw_trainable.safetensors
    saved = safetensors.torch.load_file(tmp_path / "3" / "model.safetensors")
    for n, e in zip(ema.names, ema.shadow, strict=True):
        torch.testing.assert_close(saved[n], e)
    assert torch.equal(saved["0.weight"], model[0].weight)  # frozen param saved as is
    raw = safetensors.torch.load_file(tmp_path / "3" / _ema.RAW_TRAINABLE_FILE)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(raw[n], p)

    # resume into a fresh model: raw weights back in the model, EMA restored, optimizer restored
    model2 = _toy(seed=1)
    opt2 = torch.optim.AdamW([p for p in model2.parameters() if p.requires_grad], lr=1e-2)
    ema2 = _ema.ParamEMA(model2, decay=0.8)
    step = trainer.load_checkpoint(model2, opt2, tmp_path, "cpu", ema=ema2)
    assert step == 3
    for (n, p), p2 in zip(model.named_parameters(), model2.parameters(), strict=True):
        assert torch.equal(p, p2), n
    for e, e2 in zip(ema.shadow, ema2.shadow, strict=True):
        torch.testing.assert_close(e, e2)

    # continuing from the resumed state gives the same result as continuing the original
    torch.manual_seed(7)
    _step(model, opt, ema)
    torch.manual_seed(7)
    _step(model2, opt2, ema2)
    for p, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        torch.testing.assert_close(p, p2)
    for e, e2 in zip(ema.shadow, ema2.shadow, strict=True):
        torch.testing.assert_close(e, e2)


def test_without_ema_checkpoint_is_unchanged(tmp_path):
    """pytorch_ema off: model.safetensors holds the raw weights and no raw_trainable file is written (upstream)."""
    trainer = _trainer()
    model = _toy()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    trainer.save_checkpoint(model, opt, 1, _Cfg(checkpoint_dir=tmp_path), True,
                            types.SimpleNamespace(norm_stats=None, asset_id=None))
    saved = safetensors.torch.load_file(tmp_path / "1" / "model.safetensors")
    for n, p in model.named_parameters():
        assert torch.equal(saved[n], p)
    assert not (tmp_path / "1" / _ema.RAW_TRAINABLE_FILE).exists()


def test_config_flag_defaults_off():
    from openpi.training import config as _config

    assert _config.get_config("pi05_aloha").pytorch_ema is False
