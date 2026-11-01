"""Documentation for language models (and people), shipped inside the package.

``pip install glplot`` is all it takes to give a model -- through the MCP server, a shell, or a
Python call -- an accurate picture of what GLPlot can do and how to use it:

>>> from glplot import llm
>>> print(llm.guide())                       # list the guide topics
>>> print(llm.guide("quickstart"))           # one topic, as Markdown
>>> [e.name for e in llm.search_api("heatmap")]
['imshow', 'hist2d', 'pcolormesh', 'matshow']
>>> llm.describe_api("hist2d")["signature"]  # read from the live function
>>> print(llm.example("line_plot")["code"])  # a runnable script

From a shell: ``python -m glplot.llm guide|search|describe|example|llms|doctor``.

Everything here is generated from, or tested against, the code it describes: the API index is
introspected, the tables in the guides are filled from the live registries, and every example is
executed by the test suite. Importing this package needs nothing beyond GLPlot itself.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from . import apiindex
from . import environment as _environment
from . import example_catalog as _examples
from . import guides as _topics

__all__ = [
    "agent_instructions",
    "api_categories",
    "capabilities",
    "describe_api",
    "doctor",
    "example",
    "examples",
    "guide",
    "llms_full_txt",
    "llms_txt",
    "search_api",
    "skill_markdown",
    "topics",
]


def topics(extra: Iterable[_topics.Topic] = ()) -> List[Dict[str, str]]:
    """``[{"id", "title", "summary"}, ...]`` for every guide topic, in reading order."""
    return [t.as_dict() for t in _topics.list_topics(tuple(extra))]


def guide(topic: Optional[str] = None, extra: Iterable[_topics.Topic] = ()) -> str:
    """The Markdown for one topic, or the topic index when ``topic`` is omitted.

    Raises :class:`glplot.llm.guides.UnknownTopic` (a ``KeyError``) for an unknown id; the message
    lists the valid ones.
    """
    extras = tuple(extra)
    if not topic:
        return _topics.index(extras)
    return _topics.render(_topics.find(topic, extras), extras)


def api_categories() -> List[Dict[str, Any]]:
    """Each API category with its title and the function names in it."""
    return apiindex.categories()


def search_api(
    query: str, limit: int = 8, category: Optional[str] = None
) -> List[apiindex.ApiEntry]:
    """Functions matching ``query``: a name, or words like ``"heatmap"`` / ``"second y axis"``."""
    return apiindex.search(query, limit=limit, category=category)


def describe_api(name: str) -> Dict[str, Any]:
    """Signature, summary and docstring of one function; unknown names give ``KeyError`` hints."""
    return apiindex.describe(name).as_dict(full=True)


def examples() -> List[Dict[str, str]]:
    """Name, title and description of every runnable example."""
    return _examples.list_examples()


def example(name: str) -> Dict[str, str]:
    """One example including its ``code``."""
    return _examples.get_example(name)


def capabilities() -> Dict[str, Any]:
    """Versions, optional dependencies, and the size of each part of the API."""
    return _environment.capabilities()


def doctor() -> List[Dict[str, Any]]:
    """Run the install self-checks (imports, a headless render, optional pieces)."""
    return _environment.doctor()


def llms_txt() -> str:
    """A short ``llms.txt``-style entry point: what GLPlot is and where the details are."""
    from .render import llms_txt as build

    return build()


def llms_full_txt() -> str:
    """Everything in one document: every guide topic, then the API reference by category."""
    from .render import llms_full_txt as build

    return build()


def agent_instructions() -> str:
    """Markdown for a project's ``CLAUDE.md`` / ``AGENTS.md`` that points an agent at these docs."""
    from .render import agent_instructions as build

    return build()


def skill_markdown() -> str:
    """A Claude Code skill (``SKILL.md``) that does the same."""
    from .render import skill_markdown as build

    return build()
