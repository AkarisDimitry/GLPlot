"""Assemble the single-file forms of the documentation (``llms.txt`` and ``llms-full.txt``).

``llms.txt`` follows the llmstxt.org convention: a title, a one-paragraph summary, then links.
``llms-full.txt`` is the whole guide plus an API reference in one document, for a model that can
read a file but cannot call a tool.
"""

from __future__ import annotations

from typing import Iterable, List

from . import apiindex, example_catalog, guides

__all__ = ["agent_instructions", "api_reference", "llms_full_txt", "llms_txt", "skill_markdown"]

_TAGLINE = (
    "GLPlot is a Python plotting library with a matplotlib-like API that renders on the GPU "
    "(OpenGL): millions of points stay interactive, and a figure can be saved headless with "
    "plt.savefig(). It ships a built-in MCP server so a language model can build plots, see "
    "them, analyse data, simulate differential equations, make animations (GIF, MP4, WebM) and "
    "get the equivalent Python script."
)


def llms_txt(extra: Iterable[guides.Topic] = ()) -> str:
    lines: List[str] = [
        "# GLPlot",
        "",
        f"> {_TAGLINE}",
        "",
        'Install: `pip install glplot` (add `"glplot[mcp]"` for the official MCP SDK). '
        "Import: `import glplot.pyplot as plt`. In unattended scripts call `plt.savefig(...)`, "
        "never `plt.show()`.",
        "",
        "## Read this documentation from the install itself",
        "",
        "- `python -m glplot.llm guide [topic]` -- the guides below, as Markdown",
        '- `python -m glplot.llm search "what you want"` / `describe <function>` -- the API, '
        "introspected",
        "- `python -m glplot.llm example [name]` -- runnable scripts",
        "- `python -m glplot.llm llms-full` -- this whole reference as one document",
        "- `python -m glplot.llm agent-instructions` -- a note for CLAUDE.md / AGENTS.md; "
        "`skill --write DIR` writes a Claude Code skill",
        "- `python -m glplot.mcp` -- an MCP server (tools: create_figure, add_plot, render_figure, "
        "analyze_dataset, simulate, animate, update_artist, add_control, export_script, "
        "get_guide, search_api, ...); to make a video or GIF through it read the MCP guide "
        "`mcp-animation` (`get_guide('mcp-animation')`)",
        "",
        "## Guides",
        "",
    ]
    lines.extend(f"- `{t.id}`: {t.summary}" for t in guides.list_topics(tuple(extra)))
    lines += ["", "## Examples", ""]
    lines.extend(f"- `{e['name']}`: {e['title']}" for e in example_catalog.list_examples())
    return "\n".join(lines) + "\n"


def api_reference() -> str:
    """Every indexed function: signature and first docstring sentence, grouped by category."""
    out: List[str] = ["# API reference", ""]
    by_name = {e.name: e for e in apiindex.entries()}
    for cat in apiindex.categories():
        out.append(f"## {cat['title']}")
        out.append("")
        for name in cat["names"]:  # type: ignore[union-attr]
            entry = by_name[name]
            out.append(
                f"- `{entry.name}{entry.brief}`" + (f" -- {entry.summary}" if entry.summary else "")
            )
        out.append("")
    return "\n".join(out)


def llms_full_txt(extra: Iterable[guides.Topic] = ()) -> str:
    extras = tuple(extra)
    parts = [llms_txt(extras), ""]
    for topic in guides.list_topics(extras):
        parts.append(guides.render(topic, extras))
        parts.append("")
    parts.append(api_reference())
    return "\n".join(parts).rstrip() + "\n"


_AGENT_BODY = """\
## Plotting with GLPlot

This project uses GLPlot (`import glplot.pyplot as plt`), a matplotlib-like plotting library that
draws on the GPU. Its API is close to matplotlib's but not identical, so read the documentation that
ships with the installed version instead of relying on memory:

- `python -m glplot.llm guide` lists the guides; `python -m glplot.llm guide quickstart` prints one
- `python -m glplot.llm search "second y axis"` finds functions by what they do;
  `python -m glplot.llm describe hist2d` prints a signature and docstring
- `python -m glplot.llm example surface_3d` prints a runnable script
- `python -m glplot.llm llms-full` prints all of it as one document

Rules that avoid the common failures:

- In a script you run, end with `plt.savefig("name.png")` and then look at the image. Never call
  `plt.show()` in an unattended run: it opens a window and blocks until someone closes it.
- Animation lives in `glplot.animation` (`import glplot.animation as animation`), not on `plt`.
- Inside a subplot grid draw with the axes objects (`axs[0].plot(...)`); `plt.subplot(r, c, i)` with
  a *different* shape rebuilds the grid and empties the panels already drawn.
- An option GLPlot cannot honour emits a one-time `MatplotlibCompatWarning`: read it, it lists what
  did not take effect.
- If a `glplot` MCP server is connected, its `get_guide`, `search_api`, `get_api_doc` and
  `get_example` tools return the same documentation. Through MCP you can also *animate*
  (`animate`: GIF/MP4/WebM, properties driven by time and stages; `simulate` for dynamics),
  edit what is drawn (`update_artist`), add sliders (`add_control`) and get the Python for a
  figure or animation (`export_script`): read `get_guide('mcp-animation')` before animating.
"""

_SKILL_DESCRIPTION = (
    "Make, change or debug plots, figures, animations and interactive controls with GLPlot "
    "(glplot.pyplot, a matplotlib-like GPU plotting library), or port matplotlib code to it. "
    "Use when the user mentions glplot, or when a project imports glplot."
)


def agent_instructions() -> str:
    """Markdown to paste into a project's ``CLAUDE.md`` / ``AGENTS.md`` / ``.cursor/rules``.

    A coding agent only knows this documentation exists if something tells it; this is that note,
    generated from the package so it names commands that exist in the installed version.
    """
    return _AGENT_BODY


def skill_markdown() -> str:
    """A Claude Code skill (``SKILL.md``) that sends the agent to the packaged documentation."""
    return (
        "---\n"
        "name: glplot\n"
        f"description: {_SKILL_DESCRIPTION}\n"
        "---\n\n"
        "# GLPlot\n\n" + _AGENT_BODY.split("\n", 2)[2].lstrip("\n")
    )
