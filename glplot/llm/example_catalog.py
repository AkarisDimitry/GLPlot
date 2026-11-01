"""The runnable example scripts shipped in ``glplot/llm/examples``.

Each file is a complete script: a docstring (first line = title, the rest = what to notice), then
code that ends in ``plt.savefig(...)`` so it runs with no display. They are executed by the test
suite, so an example that stops working fails CI instead of misleading a reader.
"""

from __future__ import annotations

import ast
import difflib
from functools import lru_cache
from importlib import resources
from typing import Dict, List

__all__ = ["UnknownExample", "get_example", "list_examples"]


class UnknownExample(KeyError):
    def __str__(self) -> str:
        return str(self.args[0])


@lru_cache(maxsize=1)
def _load() -> Dict[str, Dict[str, str]]:
    root = resources.files("glplot.llm") / "examples"
    found: Dict[str, Dict[str, str]] = {}
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".py") or entry.name.startswith("_"):
            continue
        code = entry.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(code)) or ""
        title, _, rest = doc.partition("\n")
        name = entry.name[: -len(".py")]
        found[name] = {
            "name": name,
            "title": title.strip().rstrip("."),
            "description": rest.strip(),
            "filename": entry.name,
            "code": code,
        }
    return found


def list_examples() -> List[Dict[str, str]]:
    """Name, title and description of every example (no code)."""
    return [{k: v for k, v in e.items() if k != "code"} for e in _load().values()]


def get_example(name: str) -> Dict[str, str]:
    """The example called ``name`` with its ``code``; unknown names suggest neighbours."""
    key = str(name).strip().lower().replace("-", "_").removesuffix(".py")
    table = _load()
    if key in table:
        return dict(table[key])
    close = difflib.get_close_matches(key, list(table), n=3, cutoff=0.4)
    words = [w for w in key.split("_") if w]
    related = [n for n in table if any(w in n for w in words)]
    hint = list(dict.fromkeys(close + related))[:4]
    tail = f" Did you mean: {', '.join(hint)}?" if hint else ""
    raise UnknownExample(f"No example {name!r}.{tail} Available: {', '.join(table)}.")
