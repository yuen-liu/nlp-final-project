"""HF steering hooks, generation, and readout for the cat lineage.

Heavy modules (model / generate / hooks) need ``requirements-gpu.txt``.
Stdlib pieces (``alpha_select``) import cleanly without torch.

Bridget usage::

    from src.steer.generate import steered_generate
    from src.steer.io import load_vector
    from src.steer.model import load_model_and_tokenizer
"""

from src.steer.alpha_select import PASS_RATE_THRESHOLD, select_alpha

__all__ = [
    "PASS_RATE_THRESHOLD",
    "select_alpha",
]
