"""``python -m glplot.llm`` -- read GLPlot's documentation from the command line.

The same material the MCP server serves, for a model (or a person) with a shell and nothing else:

    python -m glplot.llm guide                 # list the topics
    python -m glplot.llm guide quickstart      # one topic
    python -m glplot.llm search "heatmap"      # find a function by what it does
    python -m glplot.llm describe hist2d       # signature + docstring
    python -m glplot.llm examples              # runnable scripts
    python -m glplot.llm example line_plot     # print one
    python -m glplot.llm llms | llms-full      # the single-file forms
    python -m glplot.llm agent-instructions    # a note for CLAUDE.md / AGENTS.md
    python -m glplot.llm skill [--write DIR]   # a Claude Code skill
    python -m glplot.llm doctor                # does this install work?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m glplot.llm",
        description="Read GLPlot's guides, API index and examples from the command line.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    guide = sub.add_parser("guide", help="a guide topic as Markdown (no topic: list them)")
    guide.add_argument("topic", nargs="?")

    sub.add_parser("topics", help="list the guide topics")

    search = sub.add_parser("search", help="find functions by name or description")
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--category", default=None)

    describe = sub.add_parser("describe", help="signature and docstring of a function")
    describe.add_argument("name")

    sub.add_parser("categories", help="every function, grouped by category")

    sub.add_parser("examples", help="list the runnable examples")
    example = sub.add_parser("example", help="print one example's code")
    example.add_argument("name")

    sub.add_parser("llms", help="the short llms.txt")
    sub.add_parser("llms-full", help="the whole documentation as one document")
    sub.add_parser("capabilities", help="versions and feature counts, as JSON")
    sub.add_parser(
        "agent-instructions",
        help="a note to paste into CLAUDE.md / AGENTS.md so a coding agent finds these docs",
    )
    skill = sub.add_parser("skill", help="a Claude Code skill (SKILL.md) for GLPlot")
    skill.add_argument(
        "--write",
        metavar="DIR",
        help="write DIR/glplot/SKILL.md (e.g. ~/.claude/skills) instead of printing",
    )
    skill.add_argument("--force", action="store_true", help="overwrite an existing SKILL.md")
    sub.add_parser("doctor", help="check that this install can make a plot")
    return parser


def _out(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from . import (
        api_categories,
        capabilities,
        describe_api,
        doctor,
        example,
        examples,
        guide,
        llms_full_txt,
        llms_txt,
        search_api,
    )
    from .environment import format_doctor

    command = args.command or "guide"
    try:
        if command == "guide":
            _out(guide(getattr(args, "topic", None)))
        elif command == "topics":
            _out(guide())
        elif command == "search":
            hits = search_api(" ".join(args.query), limit=args.limit, category=args.category)
            if not hits:
                _out("No match. Try other words, or run: python -m glplot.llm categories")
                return 1
            for entry in hits:
                _out(f"{entry.name}{entry.brief}\n    [{entry.category}] {entry.summary}")
        elif command == "describe":
            info = describe_api(args.name)
            _out(
                f"{info['usage']}\n\ncategory: {info['category']}\n\n"
                f"{info['doc'] or '(no docstring)'}"
            )
        elif command == "categories":
            lines: List[str] = []
            for cat in api_categories():
                lines.append(f"{cat['title']}:\n  " + " ".join(cat["names"]))
            _out("\n\n".join(lines))
        elif command == "examples":
            for item in examples():
                _out(f"{item['name']:22} {item['title']}")
        elif command == "example":
            _out(example(args.name)["code"])
        elif command == "llms":
            _out(llms_txt())
        elif command == "llms-full":
            _out(llms_full_txt())
        elif command == "agent-instructions":
            from . import agent_instructions

            _out(agent_instructions())
        elif command == "skill":
            from . import skill_markdown

            if not args.write:
                _out(skill_markdown())
            else:
                target = Path(args.write).expanduser() / "glplot" / "SKILL.md"
                if target.exists() and not args.force:
                    sys.stderr.write(f"glplot.llm: {target} exists; pass --force to overwrite it\n")
                    return 1
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(skill_markdown(), encoding="utf-8")
                _out(f"Wrote {target}")
        elif command == "capabilities":
            _out(json.dumps(capabilities(), indent=2))
        elif command == "doctor":
            results = doctor()
            _out(format_doctor(results))
            return 0 if all(r["ok"] for r in results) else 1
    except KeyError as exc:
        sys.stderr.write(f"glplot.llm: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
