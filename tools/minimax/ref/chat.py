"""Chat-template rendering for MiniMax-M3.1-Flash.

The checkpoint ships ``chat_template.jinja``; it needs only ``messages``,
``tools``, ``add_generation_prompt`` and optional ``reasoning_effort``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_TEMPLATE = Path(
    os.environ.get(
        "MINIMAX_CHAT_TEMPLATE",
        r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\chat_template.jinja",
    )
)


def load_template(path: str | Path | None = None) -> str:
    p = Path(path) if path else DEFAULT_TEMPLATE
    return p.read_text(encoding="utf-8")


def apply_chat_template(
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    reasoning_effort: str | None = None,
    template: str | None = None,
) -> str:
    import jinja2

    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    # The checkpoint template calls ``| tojson(ensure_ascii=False)``; Jinja's
    # builtin filter does not take that argument (transformers supplies its own).
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=kw.get("ensure_ascii", False))
    tmpl = env.from_string(template if template is not None else load_template())
    return tmpl.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
    )
