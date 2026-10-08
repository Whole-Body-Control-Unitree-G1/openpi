"""Smoke test of scripts/train_pytorch.py with the real pi05_base weights, LoRA (JAX recipe) and EMA, on fake data.

Runs `train_loop` for a few steps (saving checkpoints), then resumes from the last checkpoint and runs a few more.

    uv run scripts/g1/smoke_train.py --weights <pytorch ckpt dir> --out <checkpoint base dir>
"""

import argparse
import dataclasses
import importlib.util
import logging
import pathlib

import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import optimizer as _optimizer

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True, help="checkpoint base dir")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    # openpi's FakeDataset makes float HWC images; the PyTorch path only converts uint8 images to CHW
    # (Observation.from_dict), as real LeRobot data arrives. Give the fake data uint8 images like real data.
    fake_getitem = _data.FakeDataset.__getitem__

    def uint8_getitem(self, index):
        item = fake_getitem(self, index)
        item["image"] = {k: np.asarray((np.asarray(v) + 1.0) * 127.5, dtype=np.uint8) for k, v in item["image"].items()}
        # FakeDataset fills bool fields with False, which masks out every image and prompt token.
        item["image_mask"] = {k: np.ones_like(np.asarray(v), dtype=bool) for k, v in item["image_mask"].items()}
        item["tokenized_prompt_mask"] = np.ones_like(np.asarray(item["tokenized_prompt_mask"]), dtype=bool)
        return item

    _data.FakeDataset.__getitem__ = uint8_getitem

    spec = importlib.util.spec_from_file_location("train_pytorch", ROOT / "scripts" / "train_pytorch.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.init_logging()

    model = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", pytorch_compile_mode=None
    )
    config = dataclasses.replace(
        _config.get_config("debug_pi05"),
        name="g1_smoke_lora_ema",
        exp_name="smoke",
        model=model,
        freeze_filter=model.get_freeze_filter(),
        pytorch_weight_path=args.weights,
        pytorch_ema=True,
        ema_decay=0.99,
        num_train_steps=args.steps,
        save_interval=1000,  # only the final step of each phase saves (each checkpoint is ~15 GB)
        log_interval=1,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-4, decay_steps=args.steps, decay_lr=1e-4),
        batch_size=2,
        num_workers=0,  # the uint8 patch above lives in this process
        checkpoint_base_dir=args.out,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
    )
    logging.info("=== phase 1: train from pi05_base")
    trainer.train_loop(config)
    logging.info("=== phase 2: resume")
    trainer.train_loop(dataclasses.replace(config, overwrite=False, resume=True, num_train_steps=args.steps + 2))


if __name__ == "__main__":
    main()
