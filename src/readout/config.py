"""Activation-readout configuration: the 3 system prompts x layer grid recorded per generation.

Working definition (README decision 5). For model M_g, eval condition s, layer l,
position p, and readout prompt x_i:

    dh_i(g, s, l, p) = h_{M_g}(x_i | s)[l, p] - h_base(x_i | s)[l, p]

The base model is read under the SAME system prompt s. That is enforced by
construction: a ReadoutConfig carries one `system_prompt`, and both M_g and
base are read with it.

Recorded per cell (see src/readout/metrics.py):
    proj  = mean_i <dh_i, v_c/|v_c|>     with a t-CI over readout prompts
    cos   = cos(mean_i dh_i, v_c)
    norm  = |mean_i dh_i|
  and, when the control lineage's cell is available (same g, s, l, p, prompts):
    proj_minus_ctrl  paired per-prompt difference, t-CI
    cos_ctrl_sub     = cos(mean dh_treat - mean dh_ctrl, v_c)   <- primary rotation metric
    cos_minus_ctrl   = cos_treat - cos_ctrl                     (also recorded)

Layer indexing matches Morgulis (layer_cosine_analysis.py L236-258) and our v_c
hooks: layer l is the OUTPUT of decoder block l (model.model.layers[l]), for
l = 0..n_layers-1. For Qwen2.5-7B that is 0..27, and v_c is injected at 2..25.
With HF `output_hidden_states`, block l is hidden_states[l + 1], since index 0
is the embeddings. Multi-hop's headline layers use the hidden_states convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.common.system_prompts import CONDITIONS, EVAL_CONDITIONS

QWEN25_7B_N_LAYERS = 28
STEER_LAYERS = tuple(range(2, QWEN25_7B_N_LAYERS - 2))  # Morgulis extract_vector.py L104

POSITIONS = ("last", "mean")
# last: final token of the rendered prompt with add_generation_prompt=True
#       (the "\n" after "<|im_start|>assistant"), i.e. Morgulis's hs[:, -1, :].
# mean: mean over the user-content tokens only (excludes system-block tokens,
#       so the system prompt's own length does not shift the average).

PROMPT_SETS = ("eval_animal", "numbers")
# eval_animal: the 50 Cloud questions (data/prompts/eval_animal.json), with no answer appended.
# numbers:     50 prompts from src/carrier/number_prompts.py with READOUT_NUMBERS_SEED.
READOUT_NUMBERS_SEED = 12345
READOUT_N_NUMBERS = 50


@dataclass(frozen=True)
class ReadoutConfig:
    system_prompt: str                     # one of EVAL_CONDITIONS; applied to M_g AND base
    lineage: str                           # e.g. "cat_steered", "control"
    generation: int                        # 1..6 (0 = teacher)
    layers: tuple[int, ...] = tuple(range(QWEN25_7B_N_LAYERS))
    positions: tuple[str, ...] = POSITIONS
    prompt_sets: tuple[str, ...] = PROMPT_SETS
    steering_vector_path: str = "artifacts/v_c/cat_blank27.pt"
    dtype: str = "bfloat16"                # forward pass; deltas computed in float32

    def __post_init__(self):
        if self.system_prompt not in EVAL_CONDITIONS:
            raise ValueError(f"system_prompt must be one of {EVAL_CONDITIONS}, got {self.system_prompt!r}")
        if not set(self.positions) <= set(POSITIONS):
            raise ValueError(f"positions must be a subset of {POSITIONS}")
        if not set(self.prompt_sets) <= set(PROMPT_SETS):
            raise ValueError(f"prompt_sets must be a subset of {PROMPT_SETS}")
        if any(not 0 <= l < QWEN25_7B_N_LAYERS for l in self.layers):
            raise ValueError("layer index out of range")

    @property
    def system_content(self) -> str | None:
        return CONDITIONS[self.system_prompt]

    def cells(self):
        """All (prompt_set, layer, position) cells this config records."""
        for ps in self.prompt_sets:
            for layer in self.layers:
                for pos in self.positions:
                    yield ps, layer, pos

    def to_dict(self) -> dict:
        return asdict(self)


def readout_grid(lineage: str, generation: int, **overrides) -> list[ReadoutConfig]:
    """One ReadoutConfig per eval system prompt: the per-generation 3 x layer grid."""
    return [ReadoutConfig(system_prompt=s, lineage=lineage, generation=generation, **overrides)
            for s in EVAL_CONDITIONS]


# --- Optional causal test (hook only; not implemented) ----------------------

@dataclass(frozen=True)
class CausalProbeConfig:
    """Present-but-gated test: add eps * v_c to M_g and to base, then compare behavior jumps.

    For each eval condition s and each eps:
        jump_Mg   = cat_rate(M_g + eps*v_c | s) - cat_rate(M_g | s)
        jump_base = cat_rate(base + eps*v_c | s) - cat_rate(base | s)
    If jump_Mg > jump_base (CI on the difference excludes 0), that is evidence that
    M_g carries the representation but it is gated. Steering is injected exactly as
    in teacher generation (same layers, all positions). eps is in units of the
    teacher's tuned alpha and must be small enough that base barely moves.
    """
    enabled: bool = False
    eps_fraction_of_alpha: tuple[float, ...] = (0.05, 0.1, 0.2)   # placeholder values, TBD
    layers: tuple[int, ...] = STEER_LAYERS
    positions: str = "all"
    eval_conditions: tuple[str, ...] = EVAL_CONDITIONS


def run_causal_probe(cfg: CausalProbeConfig, lineage: str, generation: int):
    raise NotImplementedError("causal probe is a stretch item; see CausalProbeConfig docstring")

