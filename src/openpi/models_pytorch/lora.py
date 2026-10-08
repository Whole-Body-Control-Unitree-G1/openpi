"""LoRA for the PyTorch pi0 / pi0.5 Gemma stacks, matching openpi's JAX LoRA exactly.

JAX reference: `src/openpi/models/lora.py` (`Einsum`, `FeedForward`), used by `src/openpi/models/gemma.py` for the
`gemma_*_lora` variants, and `Pi0Config.get_freeze_filter()` for what is trained.

What JAX does, and what this file reproduces:

* Attention (`lora_configs["attn"]`): LoRA on the last two axes of each einsum weight, so it is **per head**:
    q    `BTD,NDH->BTNH`    w (N, D, H)     A (N, D, r)    B (N, r, H)
    k, v `BSD,2KDH->2BSKH`  w (2, K, D, H)  A (2, K, D, r) B (2, K, r, H)   (separate for k and v, per kv head)
    out  `BTNH,NHD->BTD`    w (N, H, D)     A (N, H, r)    B (N, r, D)     (summed over heads)
  The update is scaled by `alpha / rank` (or `alpha / sqrt(rank)` with rslora).
* FFN (`lora_configs["ffn"]`): one A/B pair per matrix (gate, up, down), **not scaled** (JAX `FeedForward._dot`
  adds the LoRA product without `scaling_value`).
* A and B are both initialised N(0, 0.01) (JAX `LoRAConfig.init_fn`), not the zero-B init of the LoRA paper.
* The LoRA product is computed in the activation dtype (JAX casts `w_a`, `w_b` to `x.dtype`).
* Freezing (`get_freeze_filter`): everything under `llm` (both Gemma stacks, incl. embeddings and norms) is frozen
  except the LoRA parameters; SigLIP, the image projector and the action/time projections stay trainable.

In the PyTorch model the projections are `nn.Linear` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`); q/k/v outputs and the o_proj input are head-major (`view(..., heads, head_dim)`). `LoRALinear` keeps
the original `weight` / `bias` parameter names, so a base checkpoint (e.g. pi05_base) still loads into the model; only
the new `lora_a` / `lora_b` keys are missing.
"""

import torch
from torch import nn

from openpi.models import gemma as _gemma

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")
INIT_STD = 0.01  # JAX LoRAConfig.init_fn = normal(stddev=0.01)


class LoRALinear(nn.Linear):
    """`nn.Linear` plus a low-rank update that is block-wise per head, as in the JAX einsum LoRA.

    heads == 1 is plain LoRA. With heads == N:
      head_side="out": the output splits into N heads of size out/N; head n has A_n (r, in), B_n (out/N, r).
      head_side="in":  the input splits into N heads of size in/N; head n has A_n (r, in/N), B_n (out, r); summed.
    """

    def __init__(self, base: nn.Linear, *, rank: int, scaling: float, heads: int = 1, head_side: str = "out"):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None, device="meta")
        self.weight = base.weight  # keep the original parameters (and their names)
        self.bias = base.bias
        if head_side not in ("out", "in"):
            raise ValueError(f"head_side must be 'out' or 'in', got {head_side!r}")
        split = self.out_features if head_side == "out" else self.in_features
        if split % heads:
            raise ValueError(f"{split} features do not split into {heads} heads")
        self.rank, self.scaling, self.heads, self.head_side = rank, float(scaling), heads, head_side
        size = split // heads
        a_in = self.in_features if head_side == "out" else size
        b_out = size if head_side == "out" else self.out_features
        factory = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_a = nn.Parameter(torch.randn(heads, rank, a_in, **factory) * INIT_STD)
        self.lora_b = nn.Parameter(torch.randn(heads, b_out, rank, **factory) * INIT_STD)

    def lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lora_a.to(x.dtype), self.lora_b.to(x.dtype)
        if self.head_side == "out":
            z = torch.einsum("...i,nri->...nr", x, a)
            y = torch.einsum("...nr,nhr->...nh", z, b).flatten(-2)
        else:
            xh = x.unflatten(-1, (self.heads, -1))
            z = torch.einsum("...nh,nrh->...nr", xh, a)
            y = torch.einsum("...nr,nor->...o", z, b)
        return y * self.scaling if self.scaling != 1.0 else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.lora_delta(x)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """W + scaling * (block-wise B @ A), in the shape of `weight` (out, in)."""
        blocks = torch.einsum("nor,nri->noi", self.lora_b, self.lora_a)  # (heads, b_out, a_in)
        delta = blocks.reshape(-1, self.in_features) if self.head_side == "out" else blocks.transpose(0, 1).reshape(self.out_features, -1)
        return (self.weight.float() + self.scaling * delta).to(self.weight.dtype)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, rank={self.rank}, scaling={self.scaling}, heads={self.heads}, head_side={self.head_side}"


def apply_gemma_lora(layers: nn.ModuleList, config: _gemma.Config) -> list[nn.Parameter]:
    """Replace the attention / FFN projections of every layer with `LoRALinear`, as configured by `config.lora_configs`."""
    params: list[nn.Parameter] = []
    attn, ffn = config.lora_configs.get("attn"), config.lora_configs.get("ffn")
    heads = {"q_proj": (config.num_heads, "out"), "k_proj": (config.num_kv_heads, "out"),
             "v_proj": (config.num_kv_heads, "out"), "o_proj": (config.num_heads, "in")}
    for layer in layers:
        groups = []
        if attn is not None:
            groups.append((layer.self_attn, ATTN, attn.rank, attn.scaling_value))
        if ffn is not None:
            groups.append((layer.mlp, FFN, ffn.rank, 1.0))  # JAX FeedForward LoRA is not scaled
        for parent, names, rank, scaling in groups:
            for name in names:
                linear = getattr(parent, name)
                if isinstance(linear, LoRALinear):
                    continue
                n_heads, side = heads.get(name, (1, "out"))
                wrapped = LoRALinear(linear, rank=rank, scaling=scaling, heads=n_heads, head_side=side)
                setattr(parent, name, wrapped)
                params += [wrapped.lora_a, wrapped.lora_b]
    return params


def apply_lora(model: nn.Module, paligemma_config: _gemma.Config, action_expert_config: _gemma.Config) -> None:
    """Add LoRA to a PI0Pytorch model for every Gemma stack whose variant has `lora_configs` (the `*_lora` variants)."""
    pwe = model.paligemma_with_expert
    if paligemma_config.lora_configs:
        apply_gemma_lora(pwe.paligemma.language_model.layers, paligemma_config)
    if action_expert_config.lora_configs:
        apply_gemma_lora(pwe.gemma_expert.model.layers, action_expert_config)


def is_lora_param(name: str) -> bool:
    return name.endswith(("lora_a", "lora_b"))


def freeze_like_jax(model: nn.Module, paligemma_variant: str, action_expert_variant: str) -> None:
    """PyTorch version of `Pi0Config.get_freeze_filter()`: freeze the Gemma stacks that use LoRA, except the LoRA params.

    JAX freezes `.*llm.*` (both stacks) when the VLM uses LoRA, excluding the expert (`.*llm.*_1.*`) if the expert
    does not; or only the expert if only the expert uses LoRA. Everything else (SigLIP, projector, action/time
    projections) stays trainable.
    """
    pwe = model.paligemma_with_expert
    vlm_lora, expert_lora = "lora" in paligemma_variant, "lora" in action_expert_variant
    frozen: list[nn.Module] = []
    if vlm_lora:
        frozen += [pwe.paligemma.language_model, pwe.paligemma.lm_head]
    if expert_lora:
        frozen.append(pwe.gemma_expert)
    else:
        # The expert's lm_head is unused (pi0 decodes actions with action_out_proj) and has no JAX counterpart.
        frozen.append(pwe.gemma_expert.lm_head)
    for module in frozen:
        for name, param in module.named_parameters():
            param.requires_grad_(is_lora_param(name))


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    """Fold every LoRA update into its base weight and turn `LoRALinear` back into `nn.Linear` (for export)."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None, device="meta")
                linear.weight = nn.Parameter(child.merged_weight(), requires_grad=child.weight.requires_grad)
                linear.bias = child.bias
                setattr(parent, name, linear)


def check_pretrained_load(missing: list[str], unexpected: list[str]) -> None:
    """A base (non-LoRA) checkpoint loaded into a LoRA model may only be missing LoRA parameters."""
    bad_missing = [k for k in missing if not is_lora_param(k)]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing non-LoRA keys: {bad_missing[:10]}; unexpected: {unexpected[:10]}")


def count_params(model: nn.Module) -> dict[str, int]:
    out = {"total": 0, "trainable": 0, "lora": 0}
    for name, p in model.named_parameters():
        out["total"] += p.numel()
        out["trainable"] += p.numel() if p.requires_grad else 0
        out["lora"] += p.numel() if is_lora_param(name) else 0
    return out


__all__ = [
    "LoRALinear",
    "apply_gemma_lora",
    "apply_lora",
    "check_pretrained_load",
    "count_params",
    "freeze_like_jax",
    "is_lora_param",
    "merge_lora",
]
