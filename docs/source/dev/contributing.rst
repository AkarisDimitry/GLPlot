Contributing to GLPlot
======================

Thank you for your interest in contributing to GLPlot! This guide provides detailed instructions for reporting bugs, suggesting features, setting up your development environment, running tests, and submitting pull requests.

Code of Conduct
---------------

This project is governed by our `Code of Conduct <../../CODE_OF_CONDUCT.md>`_. By participating, you agree to uphold this code. Please report unacceptable behavior to the project maintainers.

Reporting Bugs
--------------

Before creating a bug report, check the `issue tracker <https://github.com/AkarisDimitry/GLPlot/issues>`_ to see if the problem has already been reported.

When creating a bug report, include:

**Essential Information:**

- Clear, descriptive title (e.g., "Line rendering fails with steep slopes on macOS")
- Exact reproduction steps with minimal code examples
- Expected behavior vs. actual behavior
- Screenshots or screen recordings if visually relevant

**Environment Details:**

- Python version (e.g., 3.11)
- Operating system and version
- GPU model and driver version
- GLPlot version and installation method
- Output of ``python -c "import glplot; print(glplot.__version__)"``
- Relevant dependency versions (NumPy, PyOpenGL, GLFW, etc.)

**Example Bug Report:**

::

    Title: Density visualization shows artifacts when zooming to high magnification

    Environment:
    - Python 3.11.2, macOS 14.3, Intel Iris GPU
    - GLPlot 0.1.3, installed via pip
    - PyOpenGL 3.1.6, GLFW 2.6.2, NumPy 1.24.3

    Steps to reproduce:
    1. Plot a large family of lines (>10,000)
    2. Enable density visualization
    3. Zoom in by 1000x to a specific region
    4. Pan around while zoomed

    Actual behavior: Colored artifacts appear in the visualization
    Expected behavior: Smooth density visualization without artifacts

    Code example:
    ```python
    import numpy as np
    import glplot.pyplot as plt
    # ... reproduction code ...
    ```

Suggesting Enhancements
-----------------------

Enhancement suggestions should include:

- Clear description of the proposed feature
- Use cases and examples demonstrating the benefit
- Potential implementation approach (if you have ideas)
- Comparison with similar features in other libraries
- Any potential performance implications

Suggesting API Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For API changes, provide:

- Current API usage
- Proposed API and rationale
- Backward compatibility considerations
- Migration path for existing code

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.9 or later
- Git
- pip or conda
- C compiler (for building native dependencies on some systems)

Installation for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository and navigate to it:

   .. code-block:: bash

       git clone https://github.com/AkarisDimitry/GLPlot.git
       cd GLPlot

2. Create a virtual environment (recommended):

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode with all dependencies:

   .. code-block:: bash

       pip install -e ".[dev,test]"

4. Verify the installation:

   .. code-block:: bash

       python -c "import glplot; print(f'GLPlot {glplot.__version__} installed successfully')"
       pytest --version

OpenGL and GPU Setup
~~~~~~~~~~~~~~~~~~~~~

**Linux (Debian/Ubuntu):**

.. code-block:: bash

    sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev

**macOS:**

OpenGL headers are typically included. If issues occur, ensure Xcode Command Line Tools are installed:

.. code-block:: bash

    xcode-select --install

**Windows:**

GPU drivers should provide OpenGL support. Ensure your graphics driver is up-to-date.

IDE Configuration
~~~~~~~~~~~~~~~~~~

**VSCode:**

Add to ``.vscode/settings.json``:

.. code-block:: json

    {
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true,
        "python.formatting.provider": "black",
        "editor.formatOnSave": true,
        "[python]": {
            "editor.defaultFormatter": "ms-python.python",
            "editor.formatOnSave": true
        }
    }

**PyCharm:**

- Settings → Project → Python Interpreter → Use the venv created above
- Settings → Tools → Python Integrated Tools → Default Test Runner → pytest
- Enable code inspections and format on save

Code Style Guidelines
---------------------

GLPlot follows consistent code style standards:

**PEP 8 Compliance**

- Line length: 100 characters (enforced by Black)
- Use meaningful variable and function names
- Prefer explicit over implicit (Zen of Python)

**Type Hints**

All public functions and methods must include type hints:

.. code-block:: python

    def add_line_strip(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        width: float = 1.0,
        label: str = ""
    ) -> PolylineLayer:
        """Add a continuous line strip to the plot.

        Args:
            x: X coordinates
            y: Y coordinates
            color: RGBA color tuple
            width: Line width in pixels
            label: Layer label for legend

        Returns:
            The created PolylineLayer
        """

**Docstrings**

Use Google-style docstrings for public APIs:

.. code-block:: python

    def compute_bounds(
        self,
        include_margins: bool = True
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box for all visible layers.

        Processes all layers with visible=True and aggregates their
        bounds. If include_margins is True, applies a small expansion
        to ensure visual comfort.

        Args:
            include_margins: Whether to expand bounds by 5% for margin.

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) or None if no visible layers.

        Raises:
            RuntimeError: If GPU context is not initialized.
        """

**Formatting and Linting**

GLPlot uses Black and isort for code formatting:

.. code-block:: bash

    # Format all Python files
    black glplot tests

    # Sort imports
    isort glplot tests

    # Check formatting (without modifying)
    black --check glplot tests
    isort --check-only glplot tests

Testing Requirements
--------------------

All code contributions must include appropriate tests.

Test Organization
~~~~~~~~~~~~~~~~~

Tests are organized by module in the ``tests/`` directory:

- ``test_engine.py`` - Engine initialization and core functionality
- ``test_layers.py`` - Layer creation, modification, and bounds
- ``test_pyplot_integration.py`` - pyplot API compliance
- ``test_rendering.py`` - Rendering correctness
- ``test_robustness.py`` - Edge cases and error handling

Writing New Tests
~~~~~~~~~~~~~~~~~

Test files should follow this structure:

.. code-block:: python

    """Tests for [module] functionality."""

    import numpy as np
    import pytest
    from glplot.engine import GPULinePlot
    from glplot.core.layers import ScatterLayer


    class TestLayerCreation:
        """Tests for layer instantiation and initialization."""

        def test_scatter_layer_initialization(self):
            """Scatter layer should initialize with correct defaults."""
            layer = ScatterLayer(
                x=np.array([0, 1, 2]),
                y=np.array([0, 1, 2])
            )
            assert layer.layer_type == "scatter"
            assert layer.x.shape == (3,)
            assert layer.y.shape == (3,)

        def test_scatter_layer_with_custom_colors(self):
            """Scatter layer should accept custom color arrays."""
            colors = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1]])
            layer = ScatterLayer(
                x=np.array([0, 1, 2]),
                y=np.array([0, 1, 2]),
                colors=colors
            )
            assert layer.colors.shape == (3, 4)

    @pytest.mark.parametrize("n_points", [1, 100, 10000])
    def test_scatter_rendering_with_varying_sizes(n_points):
        """Rendering should handle various data sizes."""
        plot = GPULinePlot()
        x = np.linspace(0, 1, n_points, dtype=np.float32)
        y = np.sin(2 * np.pi * x)
        plot.add_scatter(x, y)
        # Verify layer was added
        assert len(plot.scene.scatters) == 1

Test Coverage
~~~~~~~~~~~~~

- Use pytest-cov to measure coverage: ``pytest --cov=glplot --cov-report=html``
- Target ≥80% coverage for new code
- Mark slow tests with ``@pytest.mark.slow`` for CI optimization

Testing GPU Code
~~~~~~~~~~~~~~~~

GPU-dependent tests should:

1. Check for GPU availability in CI/test environments
2. Include fallback CPU verification when appropriate
3. Use fixtures to initialize GLFW/OpenGL contexts

.. code-block:: python

    @pytest.fixture
    def gpu_context():
        """Initialize GPU context for testing."""
        plot = GPULinePlot()
        yield plot
        # Cleanup
        if plot.window:
            plot.close()

Running Tests
~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

    pytest

Run specific test file:

.. code-block:: bash

    pytest tests/test_layers.py

Run with coverage report:

.. code-block:: bash

    pytest --cov=glplot --cov-report=html

Run only fast tests (skip marked slow):

.. code-block:: bash

    pytest -m "not slow"

Run with verbose output:

.. code-block:: bash

    pytest -v

Pull Request Process
--------------------

Before Submitting
~~~~~~~~~~~~~~~~~~

1. Ensure code follows style guidelines:

   .. code-block:: bash

       black glplot tests
       isort glplot tests

2. Add/update tests for your changes:

   .. code-block:: bash

       pytest tests/test_your_feature.py -v

3. Verify all tests pass:

   .. code-block:: bash

       pytest

4. Check coverage for new code:

   .. code-block:: bash

       pytest --cov=glplot --cov-report=term-missing

5. Update documentation if your changes affect public APIs

6. Create a descriptive commit message following conventions below

Commit Message Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow these conventions for commit messages:

- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Move cursor to" not "Moves cursor to"
- Limit first line to 72 characters
- Reference issues and PRs liberally

Examples:

.. code-block:: text

    Add 3D rotation support to camera controller

    Implements euler angle rotation matrices for interactive 3D
    visualization. Adds pitch, yaw, roll controls via keyboard/mouse.
    Resolves #142

    Add tests covering corner cases with extreme angles.

    Fix line rendering artifacts at high zoom levels

    Viewport-relative center projection reduces floating-point
    precision loss during extreme zooming (>10^6x). Tested up to
    sub-micron scales on large datasets.
    Fixes #301

Opening a Pull Request
~~~~~~~~~~~~~~~~~~~~~~

1. Push your branch to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

2. Open a PR on GitHub with:

   - Clear title describing the change
   - Description explaining the motivation and approach
   - Reference to any related issues (fixes #123)
   - Checklist of testing performed

3. PR template items to address:

   - What does this PR do?
   - Why is this change needed?
   - How was this tested?
   - Any breaking changes?
   - Screenshots/videos if applicable

4. Respond to review feedback:

   - Make requested changes in new commits
   - Re-request review after addressing feedback
   - Discuss disagreements constructively

PR Review Criteria
~~~~~~~~~~~~~~~~~~

Reviewers will evaluate:

- **Code Quality**: Clear, maintainable, follows project style
- **Testing**: Adequate test coverage, tests pass locally and in CI
- **Documentation**: Docstrings updated, README/guide changes if needed
- **Performance**: No regressions, GPU operations optimized
- **Backward Compatibility**: Breaking changes justified and documented

Documentation Updates
---------------------

When Documentation is Required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update documentation when:

- Adding/modifying public APIs
- Changing default behavior
- Fixing user-facing bugs
- Adding new examples or tutorials

Where to Document
~~~~~~~~~~~~~~~~~~

- **Docstrings**: All public functions and classes (module reference)
- **Guide docs**: High-level explanations (``docs/source/guide/``)
- **README.md**: Major features and usage overview
- **CHANGELOG.md**: User-facing changes for releases

Documentation Style
~~~~~~~~~~~~~~~~~~~~

- Write for users unfamiliar with GLPlot internals
- Include practical code examples
- Link to relevant API documentation
- Include output/screenshots where helpful

Example Good Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def set_colormap(
        self,
        cmap: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> None:
        """Apply a matplotlib colormap to active layer.

        Maps scalar values to colors using standard matplotlib colormaps.
        Common options: 'viridis', 'plasma', 'inferno', 'coolwarm', 'Spectral'.

        Args:
            cmap: Colormap name (see matplotlib.cm.colormaps())
            vmin: Minimum value for color scaling. If None, uses data minimum.
            vmax: Maximum value for color scaling. If None, uses data maximum.

        Example:
            >>> import glplot.pyplot as plt
            >>> plt.figure()
            >>> x = np.linspace(0, 10, 100)
            >>> plt.scatter(x, np.sin(x), c=x)
            >>> plt.set_colormap('viridis', vmin=0, vmax=10)
            >>> plt.show()

        Note:
            Colormap is applied to the currently active layer. For multiple
            layers with different colormaps, use layer.set_colormap() instead.
        """

Changelog Management
--------------------

Update ``CHANGELOG.md`` for user-visible changes:

.. code-block:: markdown

    ## [Unreleased]

    ### Added
    - Support for 3D scatter visualization with depth testing
    - Interactive axis locking with keyboard shortcuts

    ### Fixed
    - Line rendering artifacts at extreme zoom levels (#301)
    - Memory leak in density framebuffer management

    ### Changed
    - EngineOptions now accepts colormap as string name

Special Topics
--------------

Contributing GPU Code
~~~~~~~~~~~~~~~~~~~~~

When modifying GPU rendering code:

1. Document the shader algorithm with comments
2. Test on multiple GPU architectures if possible
3. Include performance benchmarks for changes
4. Consider backward compatibility with older OpenGL versions

Example shader contribution:

.. code-block:: glsl

    // Viewport-relative center projection for high-precision zooming
    // Addresses catastrophic cancellation in 32-bit float at extreme scales

    uniform vec2 viewport_center;  // Scene center (CPU double-precision)
    uniform vec2 viewport_scale;   // NDC scaling factors

    void main() {
        // Translate to viewport-relative coordinates
        vec2 rel_pos = position - viewport_center;

        // Scale to normalized device coordinates
        vec2 ndc = rel_pos * viewport_scale;

        gl_Position = vec4(ndc, 0.0, 1.0);
    }

Contributing Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before optimizing:

1. Profile to identify bottlenecks (not assumptions)
2. Measure baseline performance
3. Implement change
4. Measure improvement
5. Verify no regression with regression suite

Document performance work:

.. code-block:: python

    # Performance improvement: Reduce framebuffer copies from 2 to 1
    # Baseline: 1024x768 @ 60fps = 46.8MB/sec bandwidth
    # After: 46.8MB/sec → 23.4MB/sec (50% reduction)
    # Benchmark: tests/test_performance_benchmarks.py::test_framebuffer_throughput

Getting Help
------------

- **Questions**: Open an issue with the `question` label
- **Technical Help**: Check existing documentation and issues first
- **Discussions**: Use GitHub Discussions for design conversations
- **Email**: Contact maintainers at lombardi@fhi-berlin.mpg.de

Recognition
-----------

Contributors are recognized in:

- Project `README.md` (substantial contributions)
- Git commit history
- Release notes and CHANGELOG
- `pyproject.toml` authors list (major contributors)

Thank You!
----------

Contributing to GLPlot helps advance scientific visualization. We appreciate your effort in making GLPlot better for everyone!
