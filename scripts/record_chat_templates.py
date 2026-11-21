"""Render Qwen2.5-7B-Instruct's chat template under our three system-prompt conditions.

    python -I scripts/record_chat_templates.py <path/to/tokenizer_config.json> <revision>

This reads only tokenizer_config.json, never model weights. It renders the
template with a jinja2 sandbox configured like transformers' apply_chat_template
(trim_blocks=True, lstrip_blocks=True). Writes data/chat_templates/qwen2.5-7b-instruct.json.
Once the model is downloaded, re-check with tokenizer.apply_chat_template
(README step 4).
"""

import hashlib
import json
import sys
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.system_prompts import CONDITIONS, build_messages  # noqa: E402

USER = "Name your favorite animal using only one word."


def main(cfg_path: str, revision: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text())
    template = cfg["chat_template"]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(template)

    renders = {}
    for name in CONDITIONS:
        msgs = build_messages(name, USER)
        renders[name] = {
            "messages": msgs,
            "rendered": tmpl.render(messages=msgs, add_generation_prompt=True, tools=None),
        }

    # Sanity checks that encode our reading of the template.
    assert renders["default"]["rendered"] == renders["qwen_default_explicit"]["rendered"], \
        "omitting the system message should equal passing Qwen's default text explicitly"
    assert renders["empty"]["rendered"].startswith("<|im_start|>system\n<|im_end|>\n"), \
        "empty system content should render as an empty system block"

    out = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "revision": revision,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "renderer": "jinja2 ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)",
        "user_prompt": USER,
        "conditions": renders,
        "chat_template": template,
    }
    dest = ROOT / "data" / "chat_templates" / "qwen2.5-7b-instruct.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for name, r in renders.items():
        print(f"--- {name} ---\n{r['rendered']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
