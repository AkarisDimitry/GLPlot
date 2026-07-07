Testing Guide for GLPlot
========================

This guide covers GLPlot's testing infrastructure, how to run tests, and how to write new tests for contributions.

Testing Overview
----------------

GLPlot uses **pytest** for unit and integration testing with support for:

- Parametrized tests for data-driven testing
- Fixtures for test setup and teardown
- Coverage reporting with pytest-cov
- Performance benchmarking
- Slow test markers for CI optimization

Test Organization
-----------------

Tests are located in ``tests/`` directory and organized by feature:

.. code-block:: text

    tests/
    ├── test_engine.py              # Engine initialization and core
    ├── test_layers.py              # Layer creation and manipulation
    ├── test_options.py             # Configuration options
    ├── test_pyplot.py              # pyplot API basics
    ├── test_pyplot_integration.py  # Full pyplot workflows
    ├── test_camera_projections.py  # Camera math and transforms
    ├── test_camera_anisotropy.py   # Anisotropic camera behaviors
    ├── test_rendering.py           # Rendering correctness
    ├── test_robustness.py          # Edge cases and error handling
    ├── test_edge_cases.py          # Boundary conditions
    ├── test_api_consistency.py     # API contracts
    ├── test_export.py              # Save/print functionality
    ├── test_gallery_integration.py # Example scripts
    ├── test_geometry3d.py          # 3D geometry
    ├── test_helpers.py             # Utility functions
    ├── test_performance_benchmarks.py  # Performance regression detection
    └── test_regression.py          # Known issue regression checks

Setup and Configuration
-----------------------

Installation
~~~~~~~~~~~~

Install development dependencies:

.. code-block:: bash

    pip install -e ".[dev,test]"

This installs:

- pytest >= 7.0
- pytest-cov >= 4.0
- pytest-mock >= 3.10

Configuration
~~~~~~~~~~~~~

pytest is configured in ``pyproject.toml``:

.. code-block:: toml

    [tool.pytest.ini_options]
    addopts = "--cov=glplot --cov-report=term-missing"
    testpaths = ["tests"]
    markers = [
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    ]

Running Tests
-------------

Quick Test Run
~~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

    pytest

This runs tests and prints a coverage report.

Specific Test File
~~~~~~~~~~~~~~~~~~~

Run tests in a specific file:

.. code-block:: bash

    pytest tests/test_layers.py

Specific Test Function
~~~~~~~~~~~~~~~~~~~~~~

Run a single test:

.. code-block:: bash

    pytest tests/test_layers.py::test_scatter_layer_initialization

Specific Test Class
~~~~~~~~~~~~~~~~~~~

Run all tests in a class:

.. code-block:: bash

    pytest tests/test_layers.py::TestLayerCreation

Verbose Output
~~~~~~~~~~~~~~

Show test names and outcomes:

.. code-block:: bash

    pytest -v

Very verbose output:

.. code-block:: bash

    pytest -vv

Stop on First Failure
~~~~~~~~~~~~~~~~~~~~~

Exit immediately after first test failure:

.. code-block:: bash

    pytest -x

Stop after N failures:

.. code-block:: bash

    pytest --maxfail=3

Last Failed Tests
~~~~~~~~~~~~~~~~~

Run only the tests that failed in the last run:

.. code-block:: bash

    pytest --lf

Failed tests plus new tests:

.. code-block:: bash

    pytest --ff

Filtering Tests
~~~~~~~~~~~~~~~

Run tests by keyword:

.. code-block:: bash

    # Tests with "scatter" in name
    pytest -k "scatter"

    # Tests with "scatter" but not "3d"
    pytest -k "scatter and not 3d"

Run tests by marker:

.. code-block:: bash

    # Only fast tests
    pytest -m "not slow"

    # Only slow tests
    pytest -m "slow"

Coverage Reporting
------------------

Terminal Report
~~~~~~~~~~~~~~~

Default coverage report:

.. code-block:: bash

    pytest

This prints coverage to terminal with missing lines.

HTML Coverage Report
~~~~~~~~~~~~~~~~~~~~

Generate interactive HTML report:

.. code-block:: bash

    pytest --cov=glplot --cov-report=html

Open ``htmlcov/index.html`` in your browser to explore coverage.

Coverage by Module
~~~~~~~~~~~~~~~~~~~

Show coverage statistics:

.. code-block:: bash

    pytest --cov=glplot --cov-report=term-missing:skip-covered

Coverage Targets
~~~~~~~~~~~~~~~~

For contributions, maintain:

- **Overall coverage**: ≥80%
- **New code coverage**: ≥90%
- **Critical paths**: 100% (engine, renderers)

Writing Tests
-------------

Test Structure
~~~~~~~~~~~~~~

Follow this pattern for new tests:

.. code-block:: python

    """Tests for [feature] functionality."""

    import numpy as np
    import pytest
    from glplot.engine import GPULinePlot
    from glplot.core.layers import ScatterLayer


    class TestFeatureName:
        """Tests for [feature] behavior."""

        def test_basic_functionality(self):
            """Feature should work in basic case."""
            # Arrange: Set up test data and objects
            data = np.array([1, 2, 3])

            # Act: Call the feature
            result = some_function(data)

            # Assert: Verify behavior
            assert result is not None
            assert len(result) == 3

        def test_edge_case(self):
            """Feature should handle edge cases."""
            # ...

        @pytest.mark.slow
        def test_performance_case(self):
            """Feature should perform well with large data."""
            # ...

Naming Conventions
~~~~~~~~~~~~~~~~~~

- File: ``test_<module>.py``
- Class: ``Test<Feature>`` (e.g., ``TestScatterLayer``)
- Method: ``test_<behavior>`` (e.g., ``test_colors_applied_correctly``)

Good test names are descriptive:

- ✓ ``test_scatter_size_increases_with_values``
- ✗ ``test_scatter``
- ✓ ``test_line_rendering_fails_with_nan_coordinates``
- ✗ ``test_nan``

Using Fixtures
~~~~~~~~~~~~~~

Fixtures provide reusable test setup:

.. code-block:: python

    import pytest
    from glplot.engine import GPULinePlot


    @pytest.fixture
    def gpu_engine():
        """Create a fresh engine for testing."""
        engine = GPULinePlot()
        yield engine
        # Cleanup after test
        if engine.window:
            engine.close()


    class TestEngineFeatures:
        def test_with_fresh_engine(self, gpu_engine):
            """Test has a fresh engine instance."""
            assert gpu_engine.N == 0  # No data yet
            gpu_engine.add_line_strip(...)
            assert gpu_engine.N > 0

Parameterized Tests
~~~~~~~~~~~~~~~~~~~

Test multiple input/output combinations efficiently:

.. code-block:: python

    @pytest.mark.parametrize("n_points,expected_layers", [
        (1, 1),
        (100, 1),
        (10000, 1),
    ])
    def test_scatter_size_handling(n_points, expected_layers):
        """Rendering should handle various data sizes."""
        x = np.linspace(0, 1, n_points, dtype=np.float32)
        y = np.sin(2 * np.pi * x)

        engine = GPULinePlot()
        engine.add_scatter(x, y)

        assert len(engine.scene.scatters) == expected_layers

Testing Layer Behavior
~~~~~~~~~~~~~~~~~~~~~~

Layer testing pattern:

.. code-block:: python

    def test_line_family_layer_creation():
        """LineFamilyLayer should initialize with data."""
        ab = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
        colors = np.array([[1,0,0,1], [0,1,0,1]], dtype=np.float32)

        layer = LineFamilyLayer(
            ab=ab,
            colors=colors,
            x_range=(-10, 10)
        )

        assert layer.layer_type == "line_family"
        assert layer.ab.shape == (2, 2)
        assert layer.x_range == (-10, 10)
        assert layer.get_intrinsic_bounds() is not None

Testing Rendering
~~~~~~~~~~~~~~~~~

For GPU rendering tests:

.. code-block:: python

    @pytest.fixture
    def initialized_engine():
        """Engine with GPU context ready."""
        engine = GPULinePlot()
        engine.initialize()  # Initialize GL context
        yield engine
        engine.close()


    def test_line_rendering(initialized_engine):
        """Lines should render without errors."""
        x = np.array([0, 1, 2], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)

        # Add data
        initialized_engine.add_line_strip(x, y)

        # Trigger render (in test mode)
        initialized_engine.render_frame()

        # Verify no errors occurred
        assert initialized_engine.frame.frame_count > 0

Testing Data Flow
~~~~~~~~~~~~~~~~~

Test pipeline from API to GPU:

.. code-block:: python

    def test_pyplot_api_updates_engine():
        """pyplot.plot() should update engine scene."""
        import glplot.pyplot as plt

        plt.figure("Test")
        x = np.linspace(0, 10, 100)
        plt.plot(x, np.sin(x), "r-")

        # Access engine
        engine = plt.gcf()._engine

        # Verify data was added
        assert len(engine.scene.layers) > 0
        assert engine.scene.layers[0].layer_type == "polyline"

Testing Error Handling
~~~~~~~~~~~~~~~~~~~~~~

Verify proper error messages:

.. code-block:: python

    def test_incompatible_array_shapes_raise():
        """Function should raise with mismatched shapes."""
        x = np.array([0, 1])
        y = np.array([0, 1, 2])  # Wrong size

        with pytest.raises(ValueError, match="array shapes must match"):
            add_polyline(x, y)

Testing Edge Cases
~~~~~~~~~~~~~~~~~~

Boundary conditions and special cases:

.. code-block:: python

    @pytest.mark.parametrize("x,y", [
        ([], []),                              # Empty
        ([0], [0]),                            # Single point
        ([0, np.inf, 1], [1, 2, 3]),          # With infinity
        ([0, np.nan, 1], [1, 2, 3]),          # With NaN
    ])
    def test_boundary_conditions(x, y):
        """Function should handle edge cases gracefully."""
        # Test each case
        pass

Test Examples
-------------

Example: Testing Bounds Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class TestBoundsComputation:
        """Tests for layer bounds calculation."""

        def test_empty_scatter_has_no_bounds(self):
            """Empty scatter layer should have None bounds."""
            layer = ScatterLayer(
                x=np.array([], dtype=np.float32),
                y=np.array([], dtype=np.float32)
            )
            assert layer.get_intrinsic_bounds() is None

        def test_scatter_bounds_cover_all_points(self):
            """Bounds should encompass all points."""
            x = np.array([0, 10, 5], dtype=np.float32)
            y = np.array([1, 3, 2], dtype=np.float32)

            layer = ScatterLayer(x=x, y=y)
            bounds = layer.get_intrinsic_bounds()

            assert bounds == (0, 1, 10, 3)  # (xmin, ymin, xmax, ymax)

        @pytest.mark.parametrize("x,y,expected", [
            ([0], [0], (0, 0, 0, 0)),
            ([1, 2], [3, 4], (1, 3, 2, 4)),
            ([-5, 5], [-10, 10], (-5, -10, 5, 10)),
        ])
        def test_bounds_with_various_data(self, x, y, expected):
            """Bounds should be correct for various inputs."""
            layer = ScatterLayer(
                x=np.array(x, dtype=np.float32),
                y=np.array(y, dtype=np.float32)
            )
            assert layer.get_intrinsic_bounds() == expected

Example: Testing API Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class TestAPIConsistency:
        """Tests for API contract compliance."""

        def test_all_layers_have_required_attributes(self):
            """All layer types should have required attributes."""
            layers = [
                ScatterLayer(x=np.array([0]), y=np.array([0])),
                PolylineLayer(x=np.array([0, 1]), y=np.array([0, 1])),
                LineFamilyLayer(ab=np.array([[1, 0]]), colors=None),
            ]

            for layer in layers:
                assert hasattr(layer, 'layer_type')
                assert hasattr(layer, 'layer_id')
                assert hasattr(layer, 'style')
                assert hasattr(layer, 'dirty')
                assert hasattr(layer, 'bounds_world')

        def test_layer_style_defaults(self):
            """LayerStyle should have sensible defaults."""
            style = LayerStyle()

            assert style.visible is True
            assert style.alpha == 1.0
            assert style.zorder == 0
            assert style.line_width == 1.0

Example: Testing Color Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_rgba_color_normalization():
        """Colors should normalize to 0-1 range."""
        layer = ScatterLayer(
            x=np.array([0, 1]),
            y=np.array([0, 1]),
            colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255]])
        )

        # Verify colors are normalized if needed
        assert np.all(layer.colors <= 1.0)
        assert np.all(layer.colors >= 0.0)

Marking Tests
~~~~~~~~~~~~~

Mark tests for categorization:

.. code-block:: python

    @pytest.mark.slow
    def test_rendering_one_million_points():
        """Rendering should handle large datasets."""
        # This test takes >5 seconds
        x = np.random.random(1_000_000)
        y = np.random.random(1_000_000)
        engine = GPULinePlot()
        engine.add_scatter(x, y)
        # ...

Then run without slow tests:

.. code-block:: bash

    pytest -m "not slow"

Performance Testing
-------------------

Performance Test Template
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import time

    @pytest.mark.slow
    def test_scatter_performance_baseline():
        """Scatter rendering should maintain baseline performance."""
        n = 100_000
        x = np.random.random(n)
        y = np.random.random(n)

        engine = GPULinePlot(width=1920, height=1080)

        # Measure rendering time
        start = time.perf_counter()
        for _ in range(60):  # 60 frames
            engine.add_scatter(x, y)
            engine.render_frame()
        elapsed = time.perf_counter() - start

        # Assert minimum FPS
        fps = 60 / (elapsed / 60)
        assert fps >= 50.0, f"Only achieved {fps:.1f} FPS, need 50+"

Benchmarking
~~~~~~~~~~~~

Run performance suite:

.. code-block:: bash

    pytest tests/test_performance_benchmarks.py -v

This detects performance regressions.

Continuous Integration
----------------------

CI Test Runs
~~~~~~~~~~~~

Tests run automatically on:

- Push to pull requests
- Commits to main branch
- Manual trigger via GitHub Actions

CI Configuration (`.github/workflows/*.yml`):

- Runs on multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Tests on Linux (headless with OSMesa)
- Skips slow tests in default run
- Reports coverage to codecov

Headless Testing
~~~~~~~~~~~~~~~~

CI runs in headless environments (no X11/display):

- GLFW uses `_GLFW_OSMESA` for CPU rendering
- Tests use indirect rendering context
- No interactive windows possible

Local headless testing:

.. code-block:: bash

    export GLFW_LIBRARY=libglfw_osmesa  # Use OSMesa backend
    pytest tests/

Debugging Failed Tests
---------------------

Capture Output
~~~~~~~~~~~~~~

See print statements and logs:

.. code-block:: bash

    pytest -s tests/test_layers.py::test_specific_test

Post-mortem Debugging
~~~~~~~~~~~~~~~~~~~~~

Drop into debugger on failure:

.. code-block:: bash

    pytest --pdb tests/test_layers.py

Inspect exception:

.. code-block:: bash

    pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb

Drop to debugger on KeyboardInterrupt:

.. code-block:: bash

    pytest --pdbcls=IPython.terminal.debugger:TerminalPdb

Verbose Tracebacks
~~~~~~~~~~~~~~~~~~~

More detailed exception info:

.. code-block:: bash

    pytest --tb=long tests/test_layers.py

Other formats: ``short``, ``line``, ``native``.

Reproducible Failures
~~~~~~~~~~~~~~~~~~~~~

Run with fixed random seed:

.. code-block:: bash

    pytest --randomly-seed=12345

Useful for debugging flaky tests involving randomness.

Re-run with seed from failure output.

Best Practices
--------------

Test Independence
~~~~~~~~~~~~~~~~~

Each test should be independent:

- Don't rely on test execution order
- Clean up after yourself (use fixtures)
- Don't share state between tests

.. code-block:: python

    # Bad: Tests depend on order
    def test_1():
        global engine
        engine = GPULinePlot()

    def test_2():
        engine.add_scatter(...)  # Assumes test_1 ran

    # Good: Each test independent
    @pytest.fixture
    def engine():
        return GPULinePlot()

    def test_1(engine):
        assert engine.N == 0

    def test_2(engine):
        engine.add_scatter(...)
        assert engine.N > 0

Clear Assertions
~~~~~~~~~~~~~~~~

Make assertion messages clear:

.. code-block:: python

    # Bad: Unclear failure
    assert bounds is not None

    # Good: Clear what failed
    assert bounds is not None, f"Layer has no bounds: {layer}"
    assert bounds[2] > bounds[0], f"xmax={bounds[2]} <= xmin={bounds[0]}"

Reasonable Test Size
~~~~~~~~~~~~~~~~~~~~

Keep tests focused and fast:

- Test one thing per test
- Avoid unnecessary setup
- Use parametrization for variations

.. code-block:: python

    # Bad: Tests multiple unrelated things
    def test_scatter_layer():
        layer = ScatterLayer(...)
        assert layer is not None
        assert layer.colors is not None
        assert bounds is not None
        assert rendering_works()

    # Good: Separate concerns
    def test_scatter_layer_creation():
        layer = ScatterLayer(...)
        assert layer is not None

    def test_scatter_layer_colors():
        layer = ScatterLayer(..., colors=custom_colors)
        assert np.array_equal(layer.colors, custom_colors)

    def test_scatter_rendering():
        # Separate focused test

Use Mocks Appropriately
~~~~~~~~~~~~~~~~~~~~~~~

Mock external dependencies, not code under test:

.. code-block:: python

    # Good: Mock external dependency
    def test_export_saves_file(tmp_path, mocker):
        mock_open = mocker.patch('builtins.open', create=True)
        exporter = ExportManager()
        exporter.save_png('/tmp/test.png')
        mock_open.assert_called_once()

    # Bad: Mocking code under test defeats testing
    def test_layer_bounds(mocker):
        mocker.patch('ScatterLayer.get_intrinsic_bounds', return_value=(0,0,1,1))
        # This doesn't test the actual bounds computation!

Documentation
--------------

Test Docstrings
~~~~~~~~~~~~~~~~

Document what and why, not how:

.. code-block:: python

    def test_extreme_zoom_maintains_precision():
        """
        Verify that viewport-relative projection maintains precision
        at extreme zoom levels (10^7x). This tests the fix for issue #301
        where high zoom caused line jittering.

        The test renders the same scene at multiple zoom levels and verifies
        that rendered positions remain stable within floating-point precision.
        """

Related Documentation
---------------------

- :doc:`contributing` - Contribution workflow including testing
- :doc:`architecture` - System design and component overview
- `pytest Documentation <https://docs.pytest.org/>`_
