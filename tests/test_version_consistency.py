"""The release version is written in four places; this fails when they drift apart.

They did: 0.1.13 was published to PyPI while ``glplot.__version__`` still said 0.1.10 and
``CITATION.cff`` said 0.1.7.
"""

import re
import tomllib
from pathlib import Path

import glplot

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_package_version_matches_pyproject():
    assert glplot.__version__ == _pyproject_version()


def test_citation_version_matches_pyproject():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff has no top-level `version:`"
    assert match.group(1) == _pyproject_version()


def test_docs_release_matches_pyproject():
    text = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")
    match = re.search(r'^release\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "docs/source/conf.py has no `release = ...`"
    assert match.group(1) == _pyproject_version()


def test_changelog_has_a_section_for_this_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(
        rf"^## \[{re.escape(_pyproject_version())}\] - \d{{4}}-\d{{2}}-\d{{2}}", text, re.MULTILINE
    )
