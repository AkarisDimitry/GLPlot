"""The prose guides, shipped inside the package as Markdown.

Each ``topics/<id>.md`` starts with ``# Title``, then a ``summary: ...`` line, then the body. The
body may contain ``{{name}}`` markers; they are filled from the live code (the style presets, the
function categories, the runnable examples, the names matplotlib has that GLPlot does not) so a
table in a guide cannot disagree with the library it describes.

A caller with topics of its own -- the MCP server has workflow and file-access topics that depend
on how it was started -- passes them as ``extra`` and they are listed and served alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

__all__ = ["Topic", "UnknownTopic", "find", "index", "list_topics", "render"]

#: The order topics are listed in; any file not named here is appended alphabetically.
ORDER = (
    "overview",
    "quickstart",
    "choosing-a-plot",
    "axes-styling",
    "three-d",
    "animation",
    "controls",
    "performance",
    "export",
    "matplotlib-compat",
    "workstation",
    "mcp",
    "troubleshooting",
)


class UnknownTopic(KeyError):
    """No guide topic has this id."""

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    summary: str
    body: str

    def as_dict(self) -> Dict[str, str]:
        return {"id": self.id, "title": self.title, "summary": self.summary}


def _parse(topic_id: str, text: str) -> Topic:
    lines = text.splitlines()
    title = topic_id
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
    summary = ""
    for position, line in enumerate(lines):
        if line.startswith("summary:"):
            summary = line[len("summary:") :].strip()
            del lines[position]
            break
    return Topic(topic_id, title, summary, "\n".join(lines).strip() + "\n")


@lru_cache(maxsize=1)
def _builtin() -> Dict[str, Topic]:
    root = resources.files("glplot.llm") / "topics"
    found: Dict[str, Topic] = {}
    for entry in root.iterdir():
        if entry.name.endswith(".md"):
            topic_id = entry.name[: -len(".md")]
            found[topic_id] = _parse(topic_id, entry.read_text(encoding="utf-8"))
    ordered = [t for t in ORDER if t in found] + sorted(set(found) - set(ORDER))
    return {topic_id: found[topic_id] for topic_id in ordered}


def list_topics(extra: Iterable[Topic] = ()) -> List[Topic]:
    """Every topic, the packaged ones first (in reading order), then ``extra``."""
    topics = list(_builtin().values())
    taken = {t.id for t in topics}
    topics.extend(t for t in extra if t.id not in taken)
    return topics


def find(topic_id: str, extra: Iterable[Topic] = ()) -> Topic:
    """The topic with this id (case-insensitive; ``_`` and ``-`` are interchangeable)."""
    wanted = str(topic_id).strip().lower().replace("_", "-")
    topics = list_topics(extra)
    for topic in topics:
        if topic.id == wanted:
            return topic
    import difflib

    close = difflib.get_close_matches(wanted, [t.id for t in topics], n=3, cutoff=0.5)
    tail = f" Did you mean: {', '.join(close)}?" if close else ""
    raise UnknownTopic(
        f"No guide topic {topic_id!r}.{tail} Available: {', '.join(t.id for t in topics)}."
    )


# ----------------------------------------------------------------------------------------
# {{markers}} generated from the code
# ----------------------------------------------------------------------------------------


def _topics_list(extra: Sequence[Topic]) -> str:
    return "\n".join(f"* `{t.id}` -- {t.summary}" for t in list_topics(extra))


def _examples_list(extra: Sequence[Topic]) -> str:
    from . import example_catalog

    rows = [f"* `{e['name']}` -- {e['title']}" for e in example_catalog.list_examples()]
    rows.append(
        "\nPrint one with `python -m glplot.llm example <name>` or the `get_example` tool; "
        "each runs as-is "
        "and ends with `plt.savefig(...)`."
    )
    return "\n".join(rows)


def _categories_list(extra: Sequence[Topic]) -> str:
    from . import apiindex

    lines = []
    for cat in apiindex.categories():
        names = ", ".join(f"`{n}`" for n in cat["names"])  # type: ignore[union-attr]
        lines.append(f"**{cat['title']}** -- {names}")
    return "\n\n".join(lines)


def _styles_list(extra: Sequence[Topic]) -> str:
    from .. import pyplot

    return "\n".join(
        f"* `{key}` -- {description}" for key, _title, description in pyplot.plot_styles()
    )


def _colormaps_list(extra: Sequence[Topic]) -> str:
    from ..colormaps import CUSTOM_COLORMAPS

    return ", ".join(f"`{name}`" for name in sorted(CUSTOM_COLORMAPS))


def _unsupported_list(extra: Sequence[Topic]) -> str:
    from .. import pyplot

    return "\n".join(f"* `{name}` -- {why}" for name, why in sorted(pyplot._NOT_EXPORTED.items()))


_GENERATORS: Mapping[str, Callable[[Sequence[Topic]], str]] = {
    "topics": _topics_list,
    "examples": _examples_list,
    "categories": _categories_list,
    "styles": _styles_list,
    "colormaps": _colormaps_list,
    "unsupported": _unsupported_list,
}


def render(topic: Topic, extra: Sequence[Topic] = ()) -> str:
    """The topic's text with every ``{{marker}}`` filled in."""
    text = topic.body
    for name, generate in _GENERATORS.items():
        marker = "{{" + name + "}}"
        if marker in text:
            text = text.replace(marker, generate(extra))
    if "{{" in text and "}}" in text:
        start = text.index("{{")
        raise ValueError(
            f"topic {topic.id!r} has an unknown marker near {text[start:start + 30]!r}"
        )
    return f"# {topic.title}\n\n{text}"


def index(extra: Sequence[Topic] = ()) -> str:
    """A one-screen list of every topic and what it covers."""
    lines = ["# GLPlot guide topics", ""]
    lines.extend(f"* `{t.id}` -- {t.summary}" for t in list_topics(extra))
    return "\n".join(lines) + "\n"
