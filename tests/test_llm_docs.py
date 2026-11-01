"""The documentation a model reads (``glplot.llm``) is true, complete and runnable.

Nothing here needs the optional ``mcp`` package. The guarantees, in order of how much they matter:
every example script runs headless and writes its file; every ``plt.<name>`` a guide mentions is a
real function; the API index covers every public function with a docstring; search finds the
function a person would mean; the packaged Markdown renders with no leftover markers.
"""

from __future__ import annotations

import ast
import inspect
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import glplot.pyplot as gplt
from glplot import llm
from glplot.llm import apiindex, example_catalog, guides, render

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


# ----------------------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------------------

EXAMPLE_NAMES = [e["name"] for e in example_catalog.list_examples()]


def test_there_are_examples_and_each_has_a_title_and_description():
    assert len(EXAMPLE_NAMES) >= 12
    for item in llm.examples():
        assert item["title"] and item["description"], item["name"]
        assert item["filename"] == item["name"] + ".py"


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example_runs_headless_and_writes_its_figure(name, tmp_path, monkeypatch):
    code = llm.example(name)["code"]
    tree = ast.parse(code)
    shown = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "show"
    ]
    assert not shown, "an example must end in savefig, not block on show()"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    script = tmp_path / f"{name}.py"
    script.write_text(code)
    runpy.run_path(str(script), run_name="__main__")

    produced = {p.name for p in tmp_path.iterdir() if p.suffix in (".png", ".gif", ".pdf", ".svg")}
    assert produced, f"{name} wrote no image"
    # Every literal file name the script saves to must exist afterwards.
    for literal in re.findall(r'save(?:fig)?\(\s*["\']([\w.\-]+\.(?:png|gif|pdf|svg))["\']', code):
        assert literal in produced, (name, literal, produced)
    for path in tmp_path.iterdir():
        if path.suffix in (".png", ".gif"):
            assert path.stat().st_size > 1000, path.name


def test_unknown_example_suggests_neighbours():
    with pytest.raises(KeyError, match="line_plot"):
        llm.example("line_plt")


# ----------------------------------------------------------------------------------------
# Guides
# ----------------------------------------------------------------------------------------


def test_every_topic_renders_with_a_title_and_summary_and_no_markers():
    topics = llm.topics()
    assert len({t["id"] for t in topics}) == len(topics)
    assert {"overview", "quickstart", "animation", "mcp", "matplotlib-compat"} <= {
        t["id"] for t in topics
    }
    for topic in topics:
        assert topic["title"] and topic["summary"], topic["id"]
        text = llm.guide(topic["id"])
        assert text.startswith("# ") and "{{" not in text and "summary:" not in text, topic["id"]
        assert len(text) > 500, topic["id"]


def test_the_index_lists_every_topic():
    index = llm.guide()
    for topic in llm.topics():
        assert f"`{topic['id']}`" in index


def test_topic_ids_tolerate_case_and_underscores():
    assert llm.guide("THREE_D") == llm.guide("three-d")


def test_an_unknown_topic_says_what_exists():
    with pytest.raises(KeyError) as caught:
        llm.guide("quikstart")
    assert "quickstart" in str(caught.value) and "Available:" in str(caught.value)


def test_extra_topics_are_listed_and_served_after_the_packaged_ones():
    extra = guides.Topic("mine", "Mine", "A caller's own topic.", "Body text here.\n")
    assert [t["id"] for t in llm.topics([extra])][-1] == "mine"
    assert "Body text here." in llm.guide("mine", [extra])
    assert "mine" in llm.guide(None, [extra])


def _guide_text() -> str:
    return "\n".join(llm.guide(t["id"]) for t in llm.topics())


def test_every_plt_function_the_guides_name_exists():
    """The prose is hand-written; this is what stops it promising a function that is not there.

    A name the guide cites to say it is *absent* ("not `plt.FuncAnimation`") is allowed.
    """
    prose = re.sub(r"\bnot\b\**\s+`plt\.\w+`", "", _guide_text())
    code = "\n".join(llm.example(name)["code"] for name in EXAMPLE_NAMES)
    mentioned = set(re.findall(r"\bplt\.([A-Za-z_][A-Za-z0-9_]*)", prose + code))
    missing = sorted(name for name in mentioned if not hasattr(gplt, name))
    assert not missing, missing


def test_every_function_in_the_choosing_table_exists():
    table = llm.guide("choosing-a-plot").split("## Every function")[0]
    names = set(re.findall(r"`([a-z_0-9]+)\(", table))
    known = {e.name for e in apiindex.entries()}
    assert names and names <= known, sorted(names - known)


def test_the_matplotlib_compat_guide_lists_the_names_that_do_not_exist():
    text = llm.guide("matplotlib-compat")
    for name in gplt._NOT_EXPORTED:
        assert f"`{name}`" in text


def test_the_styles_in_the_guide_are_the_registered_ones():
    text = llm.guide("axes-styling")
    for key, _title, _description in gplt.plot_styles():
        assert f"`{key}`" in text


def test_overview_states_the_default_output_size_correctly(tmp_path):
    """The guide says figsize x 240 px; prove it, so the sentence cannot go stale."""
    from PIL import Image

    gplt.figure(figsize=(8, 5))
    gplt.plot([0, 1], [0, 1])
    target = tmp_path / "size.png"
    gplt.savefig(str(target))
    assert Image.open(target).size == (8 * 240, 5 * 240)
    assert "figsize x 240" in llm.guide("overview")


# ----------------------------------------------------------------------------------------
# API index
# ----------------------------------------------------------------------------------------


def _public_functions():
    return sorted(
        name
        for name in dir(gplt)
        if not name.startswith("_")
        and inspect.isfunction(getattr(gplt, name))
        and getattr(gplt, name).__module__ == gplt.__name__
    )


def test_the_index_covers_every_public_pyplot_function():
    indexed = {e.name for e in apiindex.entries() if e.module == gplt.__name__}
    assert set(_public_functions()) == indexed


def test_every_function_is_filed_under_a_real_category():
    other = [e.name for e in apiindex.entries() if e.category == "other"]
    assert not other, f"categorise these in glplot/llm/apiindex.py: {other}"
    for cat in llm.api_categories():
        assert cat["id"] in apiindex.CATEGORIES and cat["names"]


def test_every_public_function_has_a_docstring():
    bare = [e.name for e in apiindex.entries() if not e.doc.strip()]
    assert not bare, f"public functions without a docstring: {bare}"


def test_describe_accepts_the_spellings_a_model_uses():
    reference = llm.describe_api("scatter")
    for spelling in ("plt.scatter", "gplt.scatter", "pyplot.scatter", "SCATTER", "scatter(x, y)"):
        assert llm.describe_api(spelling)["name"] == "scatter", spelling
    assert reference["signature"].startswith("(") and reference["doc"]
    assert reference["usage"].startswith("plt.scatter(")


def test_describe_an_unknown_name_suggests_the_right_one():
    with pytest.raises(KeyError, match="scatter"):
        llm.describe_api("scater")


def test_a_class_outside_pyplot_is_indexed_with_its_import():
    entry = llm.describe_api("FuncAnimation")
    assert (
        entry["module"] == "glplot.animation" and "from glplot.animation import" in entry["usage"]
    )


@pytest.mark.parametrize(
    ("query", "expected", "within"),
    [
        ("heatmap", "imshow", 1),
        ("histogram of 2d data", "hist2d", 1),
        ("second y axis", "twinx", 5),
        ("twin axes", "twinx", 1),
        ("log scale", "xscale", 3),
        ("error bars", "errorbar", 1),
        ("shaded confidence band", "fill_between", 2),
        ("save as pdf", "savefig", 1),
        ("animate a gif", "FuncAnimation", 1),
        ("slider to change a parameter", "slider", 1),
        ("3d surface plot", "plot_surface", 1),
        ("dark theme", "plot_style", 1),
        ("violin", "violinplot", 1),
        ("reference line at y=0", "axhline", 2),
        ("streamlines", "streamplot", 1),
        ("equal aspect", "set_aspect", 1),
        ("grouped bars", "grouped_bar", 1),
        ("color by value", "scatter", 4),
        ("million points", "density", 4),
        ("polar plot", "polar", 1),
        ("colorbar", "colorbar", 1),
        ("legend", "legend", 1),
        ("subplot grid", "subplots", 1),
    ],
)
def test_search_finds_what_a_person_means(query, expected, within):
    names = [e.name for e in llm.search_api(query, limit=10)]
    assert expected in names[:within], (query, names)


def test_search_respects_limit_and_category_and_handles_nonsense():
    assert len(llm.search_api("plot", limit=3)) == 3
    assert all(e.category == "plot3d" for e in llm.search_api("plot", limit=10, category="plot3d"))
    assert llm.search_api("xyzzyq") == []
    assert llm.search_api("   ") == []


def test_brief_signatures_are_short_and_show_the_call_shape():
    for entry in apiindex.entries():
        assert len(entry.brief) <= 140, entry.name
        assert "Optional[" not in entry.brief, entry.name
    assert apiindex.describe("hist2d").brief.startswith("(x, y, bins=100")


# ----------------------------------------------------------------------------------------
# Single-file forms, capabilities, CLI
# ----------------------------------------------------------------------------------------


def test_llms_txt_is_a_short_entry_point():
    text = llm.llms_txt()
    assert text.startswith("# GLPlot") and "> " in text
    assert "python -m glplot.llm" in text and "plt.savefig" in text
    assert len(text) < 8000
    for topic in llm.topics():
        assert f"`{topic['id']}`" in text


def test_llms_full_txt_contains_every_guide_and_every_function():
    text = llm.llms_full_txt()
    for topic in llm.topics():
        assert f"# {topic['title']}" in text
    for name in _public_functions():
        assert f"`{name}(" in text, name
    assert "{{" not in text


def test_the_shipped_llms_txt_is_current():
    """The copy at the repository root (for GitHub and the docs site) matches the generator."""
    shipped = REPO_ROOT / "llms.txt"
    if not shipped.exists():
        pytest.skip("llms.txt is not in this checkout (an sdist/wheel install)")
    assert (
        shipped.read_text() == render.llms_txt()
    ), "llms.txt is stale: regenerate with `python -m glplot.llm llms > llms.txt`"


def test_capabilities_report_sizes_that_match_the_registries():
    import glplot
    from glplot.gui import recipes

    info = llm.capabilities()
    assert info["glplot"] == glplot.__version__
    assert info["api"]["style_presets"] == len(gplt.plot_styles())
    assert info["api"]["recipes"] == len(recipes.RECIPES)
    assert info["api"]["examples"] == len(EXAMPLE_NAMES)
    assert info["api"]["functions"] == len(_public_functions())


def test_doctor_passes_here_and_reports_each_check():
    results = llm.doctor()
    names = [r["check"] for r in results]
    assert "headless render" in names and "import glplot.pyplot" in names
    failed = [r for r in results if not r["ok"]]
    assert not failed, failed


class TestCommandLine:
    def run(self, capsys, *argv):
        from glplot.llm.__main__ import main

        code = main(list(argv))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_guide_lists_and_prints(self, capsys):
        code, out, _ = self.run(capsys)
        assert code == 0 and "quickstart" in out
        code, out, _ = self.run(capsys, "guide", "animation")
        assert code == 0 and "FuncAnimation" in out

    def test_search_describe_example(self, capsys):
        code, out, _ = self.run(capsys, "search", "heatmap")
        assert code == 0 and out.startswith("imshow(")
        code, out, _ = self.run(capsys, "describe", "hist2d")
        assert code == 0 and "plt.hist2d(" in out
        code, out, _ = self.run(capsys, "example", "line_plot")
        assert code == 0 and "plt.savefig" in out

    def test_errors_exit_nonzero_with_a_message(self, capsys):
        code, _, err = self.run(capsys, "guide", "nope")
        assert code == 2 and "No guide topic" in err
        code, _, err = self.run(capsys, "describe", "scater")
        assert code == 2 and "scatter" in err
        code, out, _ = self.run(capsys, "search", "xyzzyq")
        assert code == 1 and "No match" in out

    def test_single_file_forms_and_capabilities(self, capsys):
        import json

        _, out, _ = self.run(capsys, "llms")
        assert out.startswith("# GLPlot")
        _, out, _ = self.run(capsys, "llms-full")
        assert "# API reference" in out
        _, out, _ = self.run(capsys, "capabilities")
        assert json.loads(out)["api"]["functions"] > 200

    def test_it_runs_as_a_module_with_no_optional_dependency(self):
        done = subprocess.run(
            [sys.executable, "-W", "ignore", "-m", "glplot.llm", "search", "log scale"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0 and "xscale" in done.stdout, done.stderr

    def test_importing_the_package_does_not_import_the_mcp_sdk(self):
        code = (
            "import sys, glplot.llm;"
            "bad = [m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')];"
            "assert not bad, bad"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr


# ----------------------------------------------------------------------------------------
# Library fixes the guides rely on
# ----------------------------------------------------------------------------------------


class TestStats:
    def test_all_points_and_the_visible_subset(self, capsys):
        gplt.scatter([0, 1, 2, 3], [0, 1, 4, 9])
        everything = gplt.stats("all")
        assert everything["points"] == 4 and everything["x_max"] == 3.0
        gplt.xlim(0, 1.5)
        gplt.ylim(-1, 2)
        seen = gplt.stats()
        assert seen["points"] == 2 and seen["y_max"] == 1.0

    def test_nothing_plotted_is_zero_points(self, capsys):
        assert gplt.stats("all") == {"scope": "all", "points": 0}

    def test_bad_scope_is_a_value_error(self):
        with pytest.raises(ValueError, match="scope"):
            gplt.stats("everything")


class TestObjectsTheGuidesDescribe:
    """Methods the animation and controls guides tell a reader to call."""

    def test_animation_api(self):
        import glplot.animation as animation

        for name in ("pause", "resume", "to_jshtml", "save"):
            assert hasattr(animation.Animation, name), name
        assert {"pillow", "ffmpeg"} <= set(animation.writers.list())

    def test_artist_setters(self):
        from glplot.core.layers import PolylineLayer, ScatterLayer

        for name in ("set_data", "set_xdata", "set_ydata"):
            assert hasattr(PolylineLayer, name), name
        for name in ("set_offsets", "set_sizes"):
            assert hasattr(ScatterLayer, name), name

    def test_controls(self):
        gplt.figure()
        control = gplt.slider("Amplitude", 0, 1, 0.5)
        assert control.value == 0.5 and hasattr(control, "on_change")
        assert "amplitude" in gplt.controls()
        with gplt.control_group("Group"):
            assert gplt.checkbox("Flag", True).value is True
        for name in (
            "range_slider",
            "dropdown",
            "radio",
            "textbox",
            "number",
            "color_picker",
            "readout",
            "button",
            "interact",
        ):
            assert callable(getattr(gplt, name)), name


class TestAgentInstructions:
    """The note for CLAUDE.md / AGENTS.md and the Claude Code skill, for agents on a pip install."""

    def commands(self, text):
        return re.findall(r"python -m glplot\.llm ([^`\n]+)", text)

    def test_every_command_the_note_names_is_a_real_one(self):
        from glplot.llm.__main__ import build_parser

        text = llm.agent_instructions()
        found = self.commands(text)
        assert len(found) >= 4
        parser = build_parser()
        for command in found:
            arguments = [a.strip('"') for a in command.replace('"', "").split()]
            parser.parse_args(arguments)  # SystemExit if the command is not valid

    def test_the_rules_it_gives_are_the_ones_the_guides_teach(self):
        text = llm.agent_instructions()
        for fragment in (
            "plt.savefig",
            "plt.show()",
            "glplot.animation",
            "MatplotlibCompatWarning",
        ):
            assert fragment in text, fragment
        assert "get_guide" in text and "search_api" in text  # the MCP tools, when connected

    def test_the_skill_has_frontmatter_a_trigger_and_the_same_instructions(self):
        text = llm.skill_markdown()
        head, _, body = text.partition("\n---\n")
        assert head.startswith("---\nname: glplot\ndescription: ")
        description = head.split("description: ", 1)[1]
        assert "glplot" in description.lower() and len(description) > 60 and "\n" not in description
        assert body.lstrip().startswith("# GLPlot") and "python -m glplot.llm guide" in body
        assert body.count("This project uses GLPlot") == 1

    def test_llms_txt_mentions_the_agent_commands(self):
        assert "agent-instructions" in llm.llms_txt() and "skill --write" in llm.llms_txt()

    def test_the_cli_prints_and_writes(self, tmp_path, capsys):
        from glplot.llm.__main__ import main

        assert main(["agent-instructions"]) == 0
        assert capsys.readouterr().out.startswith("## Plotting with GLPlot")
        assert main(["skill", "--write", str(tmp_path)]) == 0
        written = tmp_path / "glplot" / "SKILL.md"
        assert written.read_text() == llm.skill_markdown()
        capsys.readouterr()
        assert main(["skill", "--write", str(tmp_path)]) == 1  # refuses to overwrite
        assert "--force" in capsys.readouterr().err
        written.write_text("old")
        assert main(["skill", "--write", str(tmp_path), "--force"]) == 0
        assert written.read_text() == llm.skill_markdown()
