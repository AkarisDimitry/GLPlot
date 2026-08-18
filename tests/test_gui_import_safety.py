"""Test that the pure-logic ``glplot.gui`` modules import with imgui UNAVAILABLE.

No OpenGL context and no window are created here: every check either inspects
``glplot.gui`` in-process or runs a short script in a subprocess whose ``sys.meta_path``
carries a finder that makes ``import imgui`` (and ``import imgui_bundle``) raise
ImportError.

A fact these tests deliberately encode rather than fight: ``import glplot`` DOES import
imgui, via the module-level *guarded* ``from imgui_bundle import imgui`` in
``glplot/managers/hud.py``. That guard is pre-existing upstream behaviour and is why a
missing imgui degrades to ``IMGUI_AVAILABLE = False`` instead of exploding. The property
under test is the one that actually holds and matters: importing ``glplot.gui`` and its
pure submodules never hard-fails when imgui is absent, and ``glplot/gui/__init__.py``
does not eagerly import the imgui-dependent modules (it resolves them lazily via PEP 562
``__getattr__``).
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

import glplot.gui

#: Modules that must import and work with imgui missing (CONTRACT: pure logic only).
PURE_MODULES = ["commands", "history", "datasets", "clipboard", "expressions", "mathops"]

#: Installed at the head of ``sys.meta_path`` in the subprocess: any attempt to import
#: imgui or imgui_bundle (or a submodule of either) raises ImportError, exactly as a
#: missing install would. Both names are blocked because hud.py imports the real thing
#: as ``from imgui_bundle import imgui`` -- blocking only the legacy pyimgui name would
#: leave imgui_bundle importable and defeat the "imgui is unavailable" simulation.
_BLOCKER = '''
import sys


def _is_imgui(fullname):
    return (
        fullname in ("imgui", "imgui_bundle")
        or fullname.startswith("imgui.")
        or fullname.startswith("imgui_bundle.")
    )


class _ImguiBlocker:
    """Meta-path finder that makes ``import imgui``/``imgui_bundle`` fail as if never installed."""

    def find_spec(self, fullname, path=None, target=None):
        if _is_imgui(fullname):
            raise ImportError("No module named %r (blocked by test)" % fullname)
        return None


for _name in [m for m in list(sys.modules) if _is_imgui(m)]:
    del sys.modules[_name]
sys.meta_path.insert(0, _ImguiBlocker())
'''


def _run_without_imgui(body: str) -> str:
    """Run ``body`` in a subprocess where importing imgui raises ImportError.

    Returns stdout. Raises AssertionError with the child's stderr on failure, so a
    broken import shows its traceback in the pytest report instead of a bare exit code.
    """
    script = _BLOCKER + "\n" + body
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return proc.stdout


def _is_allowed_import(node: ast.AST) -> bool:
    """True for the stdlib/``__future__`` imports the package body is allowed to hold."""
    if isinstance(node, ast.ImportFrom):
        return node.module in ("__future__", "typing")
    return all(alias.name == "typing" for alias in node.names)


class TestImguiBlocker:
    """Test the import hook itself, so the other tests cannot pass vacuously."""

    def test_blocker_makes_imgui_import_raise(self):
        """Test that ``import imgui`` raises ImportError under the hook."""
        out = _run_without_imgui(
            "try:\n"
            "    import imgui\n"
            "except ImportError:\n"
            "    print('blocked')\n"
            "else:\n"
            "    raise SystemExit('imgui was importable; the hook is broken')\n"
        )
        assert out.strip() == "blocked"

    def test_blocker_does_not_block_unrelated_modules(self):
        """Test that the hook only blocks imgui, leaving numpy importable."""
        out = _run_without_imgui("import numpy\nprint('numpy ok')\n")
        assert out.strip() == "numpy ok"


class TestPureModuleImports:
    """Test that every pure-logic gui module imports with imgui unavailable."""

    @pytest.mark.parametrize("name", PURE_MODULES)
    def test_pure_module_imports(self, name):
        """Test that each pure module imports cleanly with no imgui."""
        out = _run_without_imgui(
            f"import importlib\n"
            f"m = importlib.import_module('glplot.gui.{name}')\n"
            f"print(m.__name__)\n"
        )
        assert out.strip() == f"glplot.gui.{name}"

    def test_gui_package_imports(self):
        """Test that ``import glplot.gui`` itself succeeds with no imgui."""
        out = _run_without_imgui("import glplot.gui\nprint('ok')\n")
        assert out.strip() == "ok"

    def test_pure_modules_import_no_imgui_module_object(self):
        """Test that importing the pure modules never pulls imgui into sys.modules."""
        out = _run_without_imgui(
            "import sys, importlib\n"
            "for name in %r:\n"
            "    importlib.import_module('glplot.gui.' + name)\n"
            "def _is_imgui(m):\n"
            "    return (\n"
            "        m in ('imgui', 'imgui_bundle')\n"
            "        or m.startswith('imgui.')\n"
            "        or m.startswith('imgui_bundle.')\n"
            "    )\n"
            "print(any(_is_imgui(m) for m in sys.modules))\n"
            % (PURE_MODULES,)
        )
        assert out.strip() == "False"

    def test_hud_degrades_instead_of_raising(self):
        """Test the pre-existing guard: ``import glplot`` works, IMGUI_AVAILABLE is False."""
        out = _run_without_imgui(
            "import glplot\n"
            "from glplot.managers import hud\n"
            "print(hud.IMGUI_AVAILABLE, hud.imgui)\n"
        )
        assert out.strip() == "False None"


class TestPureModuleBehaviorWithoutImgui:
    """Test that the pure modules do not merely import, but actually work headless."""

    def test_command_queue_runs_without_imgui(self):
        """Test that CommandQueue submit/drain works with no imgui and no window."""
        out = _run_without_imgui(
            "from glplot.gui.commands import CommandQueue\n"
            "import types\n"
            "plot = types.SimpleNamespace(\n"
            "    frame=types.SimpleNamespace(dirty_scene=False, dirty_ui=False),\n"
            "    cache=types.SimpleNamespace(capture_window=(1,), refresh_requested=False),\n"
            ")\n"
            "seen = []\n"
            "q = CommandQueue()\n"
            "q.submit(lambda: seen.append(1))\n"
            "ran = q.drain(plot)\n"
            "print(ran, seen, plot.frame.dirty_scene, plot.cache.refresh_requested)\n"
        )
        assert out.strip() == "True [1] True True"

    def test_undo_stack_runs_without_imgui(self):
        """Test that history.UndoStack push/undo/redo works with no imgui."""
        out = _run_without_imgui(
            "from glplot.gui.history import Command, UndoStack\n"
            "log = []\n"
            "s = UndoStack()\n"
            "s.push(Command(label='a', do=lambda: log.append('do'),\n"
            "               undo=lambda: log.append('undo')))\n"
            "s.undo()\n"
            "s.redo()\n"
            "print(log, s.can_undo(), s.can_redo())\n"
        )
        assert out.strip() == "['do', 'undo', 'do'] True False"

    def test_datasets_runs_without_imgui(self):
        """Test that a DataSet can be built and edited with no imgui."""
        out = _run_without_imgui(
            "import numpy as np\n"
            "from glplot.gui.datasets import Column, DataSet\n"
            "ds = DataSet('d', [Column('x', np.arange(4, dtype=np.float32))])\n"
            "ds.set_cell(0, 0, 9.0)\n"
            "print(ds.n_rows(), ds.n_cols(), ds.get_cell(0, 0))\n"
        )
        assert out.strip() == "4 1 9.0"

    def test_clipboard_runs_without_imgui(self):
        """Test that clipboard.parse_table parses TSV text with no imgui."""
        out = _run_without_imgui(
            "from glplot.gui.clipboard import parse_table\n"
            "headers, data = parse_table('a\\tb\\n1\\t2\\n3\\t4\\n')\n"
            "print(headers, data.shape, data.tolist())\n"
        )
        assert out.strip() == "['a', 'b'] (2, 2) [[1.0, 2.0], [3.0, 4.0]]"

    def test_expressions_runs_without_imgui(self):
        """Test that expressions.evaluate works with no imgui."""
        out = _run_without_imgui(
            "from glplot.gui.expressions import evaluate\n"
            "print(evaluate('2 * x + 1', {'x': 3.0}))\n"
        )
        assert out.strip() == "7.0"

    def test_mathops_runs_without_imgui(self):
        """Test that mathops.definite_integral works with no imgui."""
        out = _run_without_imgui(
            "import numpy as np\n"
            "from glplot.gui.mathops import definite_integral\n"
            "x = np.linspace(0.0, 1.0, 101)\n"
            "print(round(definite_integral(x, x), 6))\n"
        )
        assert out.strip() == "0.5"


class TestGuiInitIsLazy:
    """Test that glplot/gui/__init__.py resolves its re-exports lazily (PEP 562)."""

    def test_init_declares_module_level_getattr(self):
        """Test that the package defines a module-level ``__getattr__``."""
        assert callable(glplot.gui.__getattr__)

    def test_init_has_no_eager_import_statements(self):
        """Test that the package body imports no gui submodule at module level."""
        source = pathlib.Path(glplot.gui.__file__).read_text()
        tree = ast.parse(source)
        for node in tree.body:  # module level only; the lazy import lives inside __getattr__
            assert not isinstance(node, (ast.Import, ast.ImportFrom)) or _is_allowed_import(node)

    def test_importing_gui_does_not_import_imgui_dependent_modules(self):
        """Test that ``import glplot.gui`` leaves workspace/panels/app out of sys.modules."""
        out = _run_without_imgui(
            "import sys\n"
            "import glplot.gui\n"
            "lazy = [m for m in sys.modules if m.startswith('glplot.gui.')]\n"
            "print(sorted(lazy))\n"
        )
        assert out.strip() == "[]"

    def test_lazy_names_resolve_without_imgui(self):
        """Test that the pure lazy re-exports resolve with imgui unavailable."""
        out = _run_without_imgui(
            "import glplot.gui as g\n"
            "print(g.CommandQueue.__name__, g.Command.__name__, g.UndoStack.__name__)\n"
        )
        assert out.strip() == "CommandQueue Command UndoStack"

    def test_workspace_is_not_resolved_until_accessed(self):
        """Test that ``Workspace`` is absent from the package globals until touched."""
        out = _run_without_imgui(
            "import sys, glplot.gui as g\n"
            "before = 'glplot.gui.workspace' in sys.modules\n"
            "_ = g.CommandQueue\n"
            "after = 'glplot.gui.workspace' in sys.modules\n"
            "print(before, after)\n"
        )
        assert out.strip() == "False False"

    def test_lazy_getattr_caches_resolved_value(self):
        """Test that a resolved name is written into globals() and not re-imported."""
        out = _run_without_imgui(
            "import glplot.gui as g\n"
            "print('CommandQueue' in vars(g))\n"
            "first = g.CommandQueue\n"
            "print('CommandQueue' in vars(g), g.CommandQueue is first)\n"
        )
        assert out.strip().splitlines() == ["False", "True True"]

    def test_unknown_attribute_raises_attribute_error(self):
        """Test that __getattr__ raises AttributeError, not ImportError, for junk names."""
        with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
            glplot.gui.NotAThing

    def test_dir_exposes_the_lazy_names(self):
        """Test that __dir__ advertises every lazily re-exported name."""
        names = dir(glplot.gui)
        for name in glplot.gui.__all__:
            assert name in names

    def test_all_matches_the_lazy_table(self):
        """Test that __all__ and the _LAZY resolution table agree."""
        assert sorted(glplot.gui.__all__) == sorted(glplot.gui._LAZY)
