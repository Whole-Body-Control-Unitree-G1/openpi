# openpi G1 port plan (PyTorch)

Status: **plan, for review. No code is written yet.**
Target: a fork of `Physical-Intelligence/openpi` (`main` @ `215abfb`). Work only on the PyTorch path (`src/openpi/models_pytorch/`, `scripts/train_pytorch.py`).
Goal: fine-tune pi0.5 on SONIC-teleop G1 data, with pi0.5 outputting the ScaleBFM input format, and run it with RTC through a server/client.

Port sources (pinned):

| Name | Repo @ commit | Used for |
|---|---|---|
| openpi | `Physical-Intelligence/openpi` @ `215abfb` | base |
| Chenyu | `Zhu-Chenyu/openpi`, branch `chenyu/mobilebench-v1.2` @ `66da6f2` | LoRA (`src/openpi/models_pytorch/mobilebench/lora.py`) |
| LeRobot | `huggingface/lerobot` @ `ca69a20` | per-token adaRMS, training-time RTC, guided RTC, action queue |
| Saif | `saifahmadgit/openpi_franka` @ `085ab2a` | read-only cross-check (JAX): openpi time/sign conventions for guided RTC (`models/pi0_rtc.py`), training settings. No code copied |
| PI RTC reference | `Physical-Intelligence/real-time-chunking-kinetix` | guided RTC algorithm (true VJP): `src/model.py` `realtime_action`, `get_prefix_weights` |

Rules for every step:

1. **Additive.** New code goes in new files where possible. Every change to an upstream file is small, and the step lists it.
2. **Default = upstream behaviour.** With the new feature off, outputs are bit-identical to upstream (proved by a test).
3. **Equivalence test first.** Each step has a test that must pass before the next step starts.
4. **One step = one review = one commit by the user.** Claude does not commit.
5. Keep the license headers of ported code (openpi and LeRobot are Apache-2.0).
6. **No MobileBench code.** From Chenyu's branch, take **only** `lora.py` (it imports only `torch`). Do not port, import or copy anything from `models_pytorch/mobilebench/` other than that file, or from `scripts/mobilebench/`: no dual-timescale memory, no affordance / goal / workspace decoders, no dual action heads, no episode/TBPTT trainer, no memory-keeping server. Chenyu's `model.py` (where his LoRA is wired in) is **not** ported. We wire LoRA into `PI0Pytorch` ourselves.

**Status (2026-10-08):** Steps 0, 1, 1b, 4, 5 done and tested. Next: training-time RTC (Steps 2–3), then Step 6 (G1 data config).

Order: Step 0 → 1 (+1b EMA) → 2 → 3 → 4 → 5 → 6. A **baseline** (no training-time RTC) needs Steps 0, 1, 6, and Step 5 for robot runs.

---

## Step 0 — Fork setup and golden reference

**Goal:** a working PyTorch openpi with pi05_base weights, and saved reference outputs that later steps compare against.

**Files**
- create `docs/g1/PORT_PLAN.md` (this file)
- create `CLAUDE.md` (the rules above, the source table, commands)
- create `scripts/g1/make_golden.py`
- create `tests/g1/golden/` (small `.pt` files, not large; or keep them out of git and regenerate)

**Steps**
1. Fork and clone. Make the branch (name chosen by the user).
2. `uv sync`. Copy the patched transformers files (README L207): `cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`.
3. Convert `pi05_base` JAX → PyTorch with `examples/convert_jax_model_to_pytorch.py`.
4. `make_golden.py`: build `PI0Pytorch` (pi05=True), load the weights, fix the seeds, make one fake observation (3 images, prompt, state) and fixed `noise`/`time`. Save:
   - `forward()` per-element loss
   - `sample_actions()` output (10 steps)
   - one `denoise_step()` output

**Test / done when:** two runs of `make_golden.py` give identical tensors (fp32, TF32 off).

**Note:** the transformers patch lives in `transformers_replace/` but runs from site-packages. Every later change to that folder must be copied again (Step 2 adds a helper for this).

---

## Step 1 — LoRA (port Chenyu's `lora.py`)

**Goal:** openpi's JAX LoRA recipe in PyTorch, selected by the existing config names (`paligemma_variant="gemma_2b_lora"`, `action_expert_variant="gemma_300m_lora"`). Same names as Saif's and Chenyu's configs.

Recipe (from `models/gemma.py` + `pi0_config.get_freeze_filter`, as ported by Chenyu):

| Part | Treatment |
|---|---|
| Gemma 2B (VLM language model) | frozen + LoRA r16, α16 on q,k,v,o + gate,up,down |
| Gemma 300M (action expert) | frozen + LoRA r32, α32 on the same 7 projections |
| SigLIP, multi-modal projector | fully trained |
| `action_in_proj`, `action_out_proj`, `time_mlp_in/out` | fully trained |
| LoRA init | A and B both N(0, 0.01) (openpi JAX convention, not zero-B) |

### Finding during implementation (2026-10-07): Chenyu's PyTorch LoRA is not the JAX LoRA

openpi's JAX LoRA (`models/lora.py` `Einsum`) adds the update on the last two axes of each einsum weight, so it is **per head**: q `(N, D, H)` → A `(N, D, r)`, B `(N, r, H)`; k/v per kv head; out `(N, H, D)` → A `(N, H, r)`, B `(N, r, D)`, summed over heads. FFN LoRA is one pair per matrix and **not scaled**. Chenyu's `lora.py` uses one shared A/B for q and for o (fewer parameters, no per-head blocks). We implement the JAX layout (`src/openpi/models_pytorch/lora.py`); `tests/g1/test_lora.py` checks that the LoRA and trainable parameter counts equal the JAX model's under `get_freeze_filter()`.

### LoRA: Saif vs Chenyu (what we take from each)

The LoRA **math and placement are the same** in both. Chenyu's PyTorch file is a deliberate copy of the openpi JAX LoRA that Saif uses through config names. The differences are framework and training settings:

| | Saif (`openpi_franka`) | Chenyu, JAX baseline config | Chenyu, PyTorch `lora.py` | **Ours (Step 1)** |
|---|---|---|---|---|
| Framework | JAX | JAX | PyTorch | **PyTorch** |
| LoRA code | openpi built-in (`gemma_2b_lora`, `gemma_300m_lora`) | same built-in | own 60-line `LoRALinear` | port of Chenyu's `lora.py` |
| Rank / alpha | 2B r16/α16, 300M r32/α32 | same | same | same |
| Targets | q,k,v,o + gate,up,down, every layer, both Gemmas | same | same | same |
| Init | A, B ~ N(0, 0.01) | same | same (on purpose) | same |
| Frozen | both Gemma stacks (`.*llm.*`) | same | same | same |
| Fully trained | SigLIP, projector, action/time projections | same | same, SigLIP + projector in fp32 | same; fp32 for SigLIP/projector is an option |
| Where LoRA is applied | JAX model build | JAX model build | his MobileBench `model.py` (**not ported**) | `PI0Pytorch.__init__`, driven by the existing variant names |
| Number of action experts with LoRA | 1 | 1 | 2 (MobileBench dual heads, **not ported**) | 1 |
| Learning rate | cosine, peak 2.5e-5, warmup 1–1.5k | openpi default cosine | two groups: 2.5e-5 (pretrained + LoRA), 1e-4 (new MobileBench modules) | one group (no new modules); start with Saif's 2.5e-5 cosine |
| EMA | off (`ema_decay=None`) | openpi default 0.99 | 0.99 on trainable weights | **on** (`ema_decay`, default 0.99), trainable weights only — Step 1b |
| Batch / steps | 32, 30k–50k | 16, 3k / 62.5k | — | decided in Step 6 |
| Actions | delta joints (gripper absolute) | delta (Aloha default) | — | **absolute** link poses |
| GPU | 1× A100 / H100 | 1× RTX 6000 Ada 49 GB | — | — |

**Files**
- create `src/openpi/models_pytorch/lora.py` — `LoRALinear`, `apply_gemma_lora(model, rank, alpha)` (port of Chenyu's file), plus `merge_lora(model)` for export
- modify `src/openpi/models_pytorch/pi0_pytorch.py` — in `PI0Pytorch.__init__`: if a Gemma config has `lora_configs`, call `apply_gemma_lora` on `paligemma_with_expert.paligemma.language_model` and/or `paligemma_with_expert.gemma_expert.model`, with rank/alpha from the config
- modify `scripts/train_pytorch.py` — AdamW gets only `requires_grad` params (now `model.parameters()`, ~L458); log the trainable parameter count
- create `tests/g1/test_lora.py`
- (later, Step 6) the G1 `TrainConfig` sets the `*_lora` variants

**Steps**
1. Port `LoRALinear` / `apply_gemma_lora`. Check the module paths in openpi's `gemma_pytorch.py` (Chenyu wraps the model differently).
2. Hook it into `PI0Pytorch.__init__` (config-driven, no new flags).
3. Filter the optimizer params; make sure checkpoint save/load works (the LoRA tensors are submodules, so `safetensors.torch.save_model` includes them; the model must be built with LoRA before load).
4. `merge_lora`: W ← W + (α/r)·B·A, and swap `LoRALinear` back to `nn.Linear`.

**Tests**
- With non-LoRA variants: golden outputs unchanged (bit-identical).
- With LoRA and every B set to zero: output equals golden (LoRA path adds exactly 0).
- The trainable parameter names match the JAX freeze filter (list in the test).
- One optimizer step changes only trainable params.
- `merge_lora` output equals the unmerged output (tolerance 1e-5 in fp32).

### Step 1 finding: bf16 and trainable weights

openpi's PyTorch bf16 mode (`to_bfloat16_for_selected_params`) stores SigLIP in bf16 with no fp32 copy; fine-tuning updates are then often below bf16 resolution and are lost. The JAX trainer keeps trainable params in fp32 and only frozen params in bf16 (`scripts/train.py`: "Convert frozen params to bfloat16"). `lora.trainable_to_float32` does the same; the trainer calls it after `freeze_like_jax`. Verified with a real training run: SigLIP, projector, projections and LoRA change; all frozen tensors are unchanged.

Also: openpi's `FakeDataset` sets all bool fields (image and prompt masks) to False, which hides the whole VLM input; `scripts/g1/smoke_train.py` sets them to True.

### Step 1b — EMA in the PyTorch trainer

**Why:** EMA (exponential moving average of the weights) gives smoother, usually better policies. PI uses it by default (`TrainConfig.ema_decay = 0.99`; `pi05_libero` uses 0.999). **openpi's PyTorch trainer has no EMA at all**: `ema_decay` is used only by the JAX trainer (`scripts/train_pytorch.py` has no EMA code). openpi's JAX LoRA example turns EMA off ("Turn off EMA for LoRA finetuning", `config.py` ~L696). The likely reason is memory, because the JAX EMA keeps a copy of **all** weights. In PyTorch we keep the EMA of the **trainable** weights only (LoRA + SigLIP + projections), so the cost is small. Chenyu does the same in his trainer; we write our own ~20 lines, nothing else from his trainer.

**Files**
- modify `scripts/train_pytorch.py` — if `config.ema_decay` is not None: fp32 shadow copy of the trainable params; after each `optim.step()`: `ema = d·ema + (1−d)·w` (`torch._foreach_mul_` / `_foreach_add_`); checkpoint saves both the raw weights (for resume) and the EMA weights (for serving)
- modify `src/openpi/policies/policy_config.py` (if needed) — load the EMA weights by default when present
- create `tests/g1/test_ema.py`

**Tests**
- `ema_decay=None`: training is bit-identical to the trainer without EMA.
- EMA update matches the formula on a toy model; frozen params are not tracked.
- Resume from a checkpoint restores both the raw and the EMA weights.

---

## Step 2 — Per-token adaRMS

### What adaRMS is

**RMSNorm.** Before attention and before the MLP, each token vector x (width 1024 in the action expert) is divided by its root-mean-square, then multiplied by a learned per-channel weight:
`y = x / rms(x) · (1 + w)`.

**Adaptive RMSNorm (adaRMS).** The weight is not fixed. A small dense layer reads a **condition vector** c and outputs three vectors:
`scale, shift, gate = Dense(c)`
`y = x / rms(x) · (1 + scale) + shift`.
The `gate` multiplies the output of the attention / MLP block before it is added back to the residual stream. In pi0.5 the condition c is the **flow time embedding** (sinusoidal embedding of t → `time_mlp_in` → SiLU → `time_mlp_out` → SiLU). So every layer of the action expert knows "how noisy the actions are now" and adjusts its normalisation. This is the same idea as adaLN-Zero in DiT. The dense layer starts at zero, so at init it is a plain RMSNorm with gate 0.

Only the **action expert** uses adaRMS (`use_adarms=[False, True]`). The 2B VLM uses plain RMSNorm.

**Why it must become per-token.** Today one sample has one t, so c has shape (B, D) and the same scale/shift/gate is used for all 50 action tokens. Training-time RTC gives the **clean prefix tokens t = 0** and the other tokens the sampled t. So each token needs its own c: shape (B, H, D). openpi's norm always does `modulation.unsqueeze(1)`, which is wrong for a (B, H, D) condition. LeRobot's norm handles both (`policies/pi_gemma.py` L120–140).

**Goal:** accept `timestep` of shape (B,) or (B, H) everywhere, with identical results when all H entries are equal.

**Files**
- modify `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` — the adaptive norm `forward(x, cond)` (~L73–100): accept cond (B, D) or (B, T, D); add the token axis only for (B, D); check T matches. Port LeRobot `pi_gemma.py` L120–140.
- modify `src/openpi/models_pytorch/pi0_pytorch.py` — `embed_suffix` (~L238–315): accept `timestep` (B,) or (B, H). `create_sinusoidal_pos_embedding` (~L25) must work on a (B, H) input (flatten → embed → reshape). The time MLP then outputs (B, H, D).
- check `src/openpi/models_pytorch/gemma_pytorch.py` — the cond is passed through unchanged (L157–264); fix only if a shape assumption breaks.
- create `scripts/g1/sync_transformers_patch.sh` — copies `transformers_replace/*` into the venv
- create `tests/g1/test_adarms_per_token.py`

**Tests**
- (B,) time: golden outputs unchanged.
- (B, H) time with all entries = t: equal to the (B,) result (tolerance ~1e-6 in fp32).
- (B, H) time with different values: runs, and changing token k's time changes token k's output.

---

## Step 3 — Training-time RTC

**Paper:** Black, Ren, Equi, Levine, "Training-time action conditioning for efficient real-time chunking", arXiv 2512.05964.
**Idea:** during training, sample a delay d ∈ [0, max_delay] per example. The first d actions are given **clean** (time 0) as a prefix, and the loss is computed only on the other actions. At run time, the frozen prefix (the actions already committed, d + ℓ in our system) is fed in clean, so no extra guidance pass is needed.

**Port from LeRobot**
- `policies/pi05/modeling_pi05.py` L131–160: `_sample_training_rtc_prefix_mask`, `_reduce_training_rtc_loss`
- `policies/pi05/modeling_pi05.py` L80–128: `_prepare_trained_rtc_prefix` (inference side, validation)
- `policies/common/flow_matching.py` L188+: `make_flow_matching_inputs(..., prefix_mask)` and `euler_integrate(..., hard_prefix, hard_prefix_mask)`
- Both repos use the same convention: `x_t = t·noise + (1−t)·actions`, `u = noise − actions`, sampling from t = 1 to 0 (openpi `pi0_pytorch.py` forward and `sample_actions`). So the clean end is **t = 0**.

**Files**
- create `src/openpi/models_pytorch/rtc_training.py` — the three helpers + `make_flow_matching_inputs`
- modify `src/openpi/models_pytorch/pi0_pytorch.py`
  - `forward`: if `rtc_training_max_delay > 0`, sample the prefix mask; per-token time (prefix → 0); x_t = clean actions on the prefix; return the loss with the prefix masked out
  - `sample_actions`: optional `prev_chunk` + `inference_delay`; in the Euler loop, overwrite the prefix with the clean previous actions and give those tokens time 0 at every step
- modify `src/openpi/models/pi0_config.py` — add `rtc_training_max_delay: int = 0`
- modify `scripts/train_pytorch.py` — loss reduction over the postfix only (today it averages the per-element loss returned by `forward`)
- create `tests/g1/test_rtc_training.py`

**Tests**
- `rtc_training_max_delay = 0`: loss and samples equal golden (bit-identical).
- Prefix positions: x_t equals the clean actions, time = 0, loss contribution = 0.
- **LeRobot cross-check:** load the same pi05_base weights in LeRobot's pi05 and in our model; same batch, noise, time and prefix mask → loss within bf16 tolerance. Same for one trained-mode sample.
- Inference with `inference_delay = 0`: equals vanilla `sample_actions`.

**Out of scope here:** choosing max_delay (later: ≥ d + ℓ + margin).

---

## Step 4 — Guided inference-time RTC

**Paper:** Black, Galliker, Levine, "Real-Time Execution of Action Chunking Flow Policies", arXiv 2506.07339; PI blog `pi.website/research/real_time_chunking`.
**Idea:** no retraining. At each denoising step, add a guidance term (vector-Jacobian product) that pulls the predicted clean chunk toward the previous chunk on the overlap, weighted by the soft mask (1 on the frozen d actions, exponential decay, 0 on the last s). Works on any checkpoint, including the Step 1 baseline.

**Sources (compared 2026-10-07; see "Guided RTC: source comparison" below)**
- **Algorithm: PI reference** `real-time-chunking-kinetix` `src/model.py` — `realtime_action` (L219–250, true VJP through the denoiser) and `get_prefix_weights` (L40). Ported to PyTorch.
- **Convention cross-check: Saif** `src/openpi/models/pi0_rtc.py` (JAX) — the same algorithm already mapped to openpi's time (t=1 noise) and velocity sign (`v − w·corr`). Read-only reference, no code copied.
- **Structure / config: LeRobot** `policies/rtc/configuration_rtc.py` (execution horizon, β, schedule) and the `RTCProcessor.denoise_step` wrapper shape (`policies/rtc/modeling_rtc.py` L122–250).

**Guided RTC: source comparison**

| Part | PI reference | Saif (JAX) | LeRobot (PyTorch) |
|---|---|---|---|
| Correction | VJP through the network | same VJP | **err only**: `v_t` is computed before `x_t.requires_grad_(True)`, so the Jacobian is dropped (checked with a toy torch test) |
| Guidance weight, sign, prefix weights | reference | same (after time/sign conversion) | same |
| Soft-region end | `prefix_attention_horizon` (= H − s) | all unplayed actions | `execution_horizon` (default 10) |
| Defaults | β 5, exp, 5 steps | β 5, exp, 10 steps | β 10, linear, 10 steps |

Decision: default = true VJP (the paper's ΠGDM). Option `use_vjp=False` = LeRobot's identity-Jacobian approximation (no backward pass, faster). Measure both.

**Files**
- create `src/openpi/models_pytorch/rtc_guided.py` — config dataclass (`execution_horizon`, `max_guidance_weight`, `schedule`, `use_vjp`), `get_prefix_weights`, guided Euler loop that wraps `PI0Pytorch.denoise_step` (gradients w.r.t. x_t only, with `torch.enable_grad()`; weights stay frozen)
- ~~modify `pi0_pytorch.py`~~ not needed: `rtc_guided.sample_actions_guided(model, ...)` builds the prefix cache and wraps `PI0Pytorch.denoise_step`; the Step 5 server calls it directly (upstream file unchanged)
- create `tests/g1/test_rtc_guided.py`

**Result (2026-10-07):** 37 tests pass. Weights equal kinetix (all schedules) and LeRobot (linear); VJP equals finite differences (1e-6, fp64); no-VJP equals LeRobot's formula; zero weights reproduce the golden sample bit-for-bit; on pi05_base the first 8 actions move to the previous chunk (|diff| 0.223 → 0.027 VJP, 0.004 no-VJP). Latency in strict-fp32 test mode: vanilla 800 ms, no-VJP 809 ms, VJP 1285 ms (serving latency is measured in Step 5).

**Tests**
- No previous chunk, or schedule ZEROS: equals vanilla `sample_actions` (bit-identical with the same noise).
- `get_prefix_weights` equals the kinetix reference for several (d, s, H).
- With a previous chunk and β > 0: the first d actions are closer to the previous chunk than without guidance.
- `use_vjp=False`: equals LeRobot's `RTCProcessor.denoise_step` on the same inputs (bf16 tolerance).
- `use_vjp=True`: the correction equals a finite-difference VJP on a small model; on a linear toy denoiser it equals `err − t·err·J`.
- Latency report: vanilla vs guided (expected ≈ +25–30% from the backward pass).

---

## Step 5 — Server and client

**Goal:** remote inference with RTC. The robot-side client keeps a 50 Hz action queue and handles the timing (d from latency, ℓ from ScaleBFM's lag).

**Protocol** (LeRobot semantics over openpi's websocket; LeRobot itself has RTC only in-process in `lerobot-rollout`, and its gRPC `async_inference/` has no RTC):
- LeRobot API being mirrored: `predict_action_chunk(obs, inference_delay=d, prev_chunk_left_over=...)`; queue `policies/rtc/action_queue.py` keeps `original_actions` (model space, normalized) and `processed_actions` (robot units); `get_left_over()` returns the model-space remainder.
- server → client: `actions` (H, D, absolute joint targets in robot units) + `server_timing`
- client → server: observation (incl. current state) + `prev_chunk_left_over` (**absolute**, robot units, from the queue; omitted on the first call) + `inference_delay` (= d + ℓ, in steps) + `rtc_mode` (`none` / `guided` / `trained`)
- the server is stateless. Because the actions are **relative** (Step 6), the server re-anchors the leftover to the new state and normalizes it before it becomes the prefix: port of LeRobot `policies/rtc/relative.py` `reanchor_relative_rtc_prefix` (= `DeltaActions` + `Normalize` of openpi's input transforms)

**Files**
- create `src/openpi/policies/policy_rtc.py` — `RTCPolicy(Policy)`: pops the RTC keys, re-anchors + normalizes `prev_chunk_left_over` with the same input transforms as training, passes it and `inference_delay` to the guided or trained sampler. `policy.py` stays unchanged.
- test: re-anchored prefix equals LeRobot `reanchor_relative_rtc_prefix` on the same inputs
- create `scripts/serve_policy_rtc.py` — same CLI as `scripts/serve_policy.py` + RTC options (β, schedule, num_steps)
- create `packages/openpi-client/src/openpi_client/rtc_action_queue.py` — port of LeRobot `policies/rtc/action_queue.py` (281 lines) + `latency_tracker.py` (72 lines). Additions:
  - each chunk stamped with its observation time t_obs
  - read index `j = (now − t_obs)/Δt + ℓ` (ℓ = ScaleBFM lag, a fixed parameter)
  - `inference_delay = d + ℓ`, d from the latency tracker (max over recent inferences)
  - request a new chunk every s steps; check `s + d + ℓ + K ≤ H`
- create `packages/openpi-client/src/openpi_client/rtc_client.py` — the loop: get obs → send → merge → 50 Hz output callback
- create `examples/g1/fake_robot_client.py` — a fake 50 Hz robot (no hardware) for tests; the real ScaleBridge adapter comes later
- create `tests/g1/test_action_queue.py`

**Tests**
- Action queue unit tests: correct index with d and ℓ; correct frozen prefix; no gap when the rule holds; raises when the rule does not hold.
- End-to-end on localhost with the fake robot: `rtc_mode=none` equals the stock `serve_policy.py` path; `guided` and `trained` run at 50 Hz with no empty queue at the measured latency.
- Plots: commanded vs ideal trajectory at chunk boundaries (same plot as the explainer simulator).

**Out of scope:** the ScaleBridge `motion_tracking_vla` env (separate repo, separate plan).

---

**Result (2026-10-08):**
- 5a `policy_rtc.RTCPolicy` + `scripts/serve_policy_rtc.py`: re-anchors the previous chunk with the training transforms (round trip 1e-5); pi05_aloha on pi05_base, first 8 actions |new − old| 0.261 → 0.016 rad, hand-over jump 0.469 → 0.088 rad.
- 5b `openpi_client.rtc_action_queue` / `rtc_client`: tick-stamped chunks, target for now + l, frozen prefix d + l, LeRobot-style warm-up (first 2 real requests compile the server, are not executed, not timed), `reset()`; 11 tests.
- 5c `examples/g1/fake_robot_client.py` + `scripts/g1/run_e2e.sh` (server + 50 Hz fake robot, localhost): s = 20, l = 5. Latency median vanilla 88 ms, RTC no-VJP 109 ms, RTC VJP 219 ms (an earlier run on a busy GPU: 276 / 344 / 595 ms). Largest hand-over jump: no RTC 0.77 rad, no-VJP 0.19, VJP 0.053; no starvation.
- Upstream changes: `models/model.py` `load_pytorch` builds on the GPU (CPU init took ~6 min); `serving/websocket_policy_server.py` runs inference in a worker thread (a long first request no longer misses the client's keepalive pings).
- Not done: safety-stop detection (user: restart the process after damping); Saif-style server warm-up with a G1 fake observation (Step 6).

## Step 6 — G1 data config

**Goal:** train on SONIC-teleop G1 data. pi0.5 predicts **joint targets**; the client converts them to ScaleBFM link poses at run time, so the ScaleBFM mode is a run-time choice.

**Why joints, not link poses (decided 2026-10-07)**
- openpi: "pi0 models are trained on delta actions (relative to the first state in each action chunk) … gripper actions are always absolute" (`training/config.py` L326). All openpi joint recipes use `DeltaActions` (ALOHA L232, DROID joint-position L404, UR5 example); so do Saif and Chenyu.
- `DeltaActions` subtracts `state[..., :n]` from `actions[..., :n]` (`transforms.py` L204–222), so the action must have the **same layout as the state**. Joints do; link poses do not.
- From joints, FK gives all 14 ScaleBFM links, so mode 2 / 6 / 7 needs no retraining.

**Interface**
- state (31): 29 body joints (ScaleBFM joint order) + 2 Dex1 gripper positions — written into the prompt as text
- action (31): the same layout. `DeltaActions(make_bool_mask(29, -2))`: 29 joints relative to the chunk's first state, 2 grippers absolute. `AbsoluteActions` on the output.
- images: `ego_view` → `base_0_rgb`, `left_wrist` → `left_wrist_0_rgb`, `right_wrist` → `right_wrist_0_rgb`
- prompt: episode task (`prompt_from_task=True`)
- fps 50, action_horizon 50
- labels: measured joints (no FK at training time)

**Run time (client, Step 5 + ScaleBridge)**
- absolute joint targets → FK (ScaleBFM `g1_29dof.xml`) → link poses of the chosen mode, frames `[0..4, K]`
- pelvis pose: feet anchored at their start positions, FK up through the legs (same method as the motion clips)
- default mode 7 (closest to the demonstrations); modes 2 / 6 as experiments
- RTC with relative actions: re-anchor the previous chunk to the new state before it is used as the prefix (port of LeRobot `policies/rtc/relative.py` `reanchor_relative_rtc_prefix`) — add to Step 5

**Files**
- create `scripts/g1/convert_sonic_to_g1.py` — SONIC LeRobot v2.1 → pi0.5 LeRobot v2.1; joint reorder (GR00T robot-model order → ScaleBFM order); state = action layout; trims idle start/end
- create `scripts/g1/check_conversion.py` — FK of one converted episode → ScaleBFM motion `.npz` (pelvis from foot-anchored FK), for the MuJoCo replay test in modes 2 / 6 / 7
- create `src/openpi/policies/g1_policy.py` — `G1Inputs` / `G1Outputs` (key mapping, image slots, keep the first 31 action dims)
- modify `src/openpi/training/config.py` — `LeRobotG1DataConfig` (with `DeltaActions`) + `TrainConfig`s:
  - `pi05_g1_stand_lora` (baseline: LoRA + EMA, no training-time RTC)
  - `pi05_g1_stand_lora_rtc` (same + `rtc_training_max_delay`)
- run `scripts/compute_norm_stats.py --config-name pi05_g1_stand_lora` (stats in delta space)
- create `tests/g1/test_g1_policy.py`

**Tests**
- `DeltaActions` → `AbsoluteActions` round trip returns the recorded joints.
- Foot-anchored FK of recorded joints matches the robot's recorded pelvis orientation (IMU) and plausible height.
- MuJoCo replay of 5 converted episodes with ScaleBFM: hand error per mode (2 / 6 / 7).
- Data loader: one batch has the expected shapes; the state prompt fits `max_token_len`.
- 200-step smoke training run: the loss goes down.

---

## Open questions (for the user)

1. Branch name, fork location, and which GPU / cluster for training.
2. Should Step 5's client live in `openpi-client` (here) or in ScaleBridge?
3. Baseline first: run Steps 0, 1, 6 (+5) before Steps 2–4?
