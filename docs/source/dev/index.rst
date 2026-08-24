Developer Guide
===============

This section contains comprehensive guides for developers contributing to or extending GLPlot.

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   contributing
   architecture
   testing

Quick Links
-----------

- **Contributing**: Start here if you want to contribute code, report bugs, or suggest features
- **Architecture**: Understand GLPlot's design, component structure, and rendering pipeline
- **Testing**: Learn how to run tests, write new tests, and validate changes

Getting Started
---------------

New contributors should:

1. Read :doc:`contributing` for setup instructions and contribution workflow
2. Explore :doc:`architecture` to understand the system design
3. Review :doc:`testing` to understand how to validate changes
4. Check the `main README <../../README.md>`_ for high-level overview
5. Look at examples in the `examples/` directory to understand the API

Key Resources
-------------

- `GitHub Repository <https://github.com/AkarisDimitry/GLPlot>`_
- `Issue Tracker <https://github.com/AkarisDimitry/GLPlot/issues>`_
- `Code of Conduct <../../CODE_OF_CONDUCT.md>`_
- `Mathematical Formulation <../../ARCHITECTURE.md>`_

Development Environment
-----------------------

Quick setup:

.. code-block:: bash

    git clone https://github.com/AkarisDimitry/GLPlot.git
    cd GLPlot
    pip install -e ".[dev,test]"
    pytest  # Run tests to verify setup

Common Tasks
------------

**Running tests:**

.. code-block:: bash

    pytest                              # Run all tests
    pytest -m "not slow"               # Skip slow tests
    pytest --cov=glplot               # With coverage report

**Code formatting:**

.. code-block:: bash

    black glplot tests
    isort glplot tests

**Building documentation:**

.. code-block:: bash

    cd docs
    make html

**Interactive development:**

.. code-block:: python

    import glplot.pyplot as plt
    import numpy as np

    plt.figure("Demo")
    x = np.linspace(0, 10, 100)
    plt.plot(x, np.sin(x), "r-", label="sin(x)")
    plt.show()

Architecture Overview
---------------------

GLPlot's high-level architecture:

::

    Application (pyplot API)
           ↓
    Scene Graph (Layers)
           ↓
    Rendering Engine (GPU dispatch)
           ↓
    GPU Pipeline (Shaders, FBOs)
           ↓
    Platform (GLFW, OpenGL)

Read :doc:`architecture` for detailed breakdown of each component.

Testing Strategy
----------------

GLPlot maintains comprehensive test coverage:

- **Unit tests**: Individual components (layers, options, etc.)
- **Integration tests**: Subsystem interactions (pyplot workflows)
- **Regression tests**: Known issues don't resurface
- **Performance tests**: No degradation with large datasets

See :doc:`testing` for running tests and writing new ones.

Code Style
----------

- **Language**: Python 3.9+
- **Line length**: 100 characters (Black)
- **Type hints**: All public APIs
- **Docstrings**: Google-style for public functions
- **Imports**: Organized with isort

Run formatters before committing:

.. code-block:: bash

    black glplot tests
    isort glplot tests

Common Development Workflows
-----------------------------

Adding a New Feature
~~~~~~~~~~~~~~~~~~~~~

1. Create a branch: ``git checkout -b feature/my-feature``
2. Write tests first (test-driven development)
3. Implement the feature
4. Update documentation
5. Run full test suite: ``pytest``
6. Format code: ``black glplot tests && isort glplot tests``
7. Commit with descriptive message
8. Push and open pull request

Debugging Issues
~~~~~~~~~~~~~~~~

1. Check existing issues on GitHub
2. Reproduce in minimal example
3. Enable verbose logging if available
4. Run specific test: ``pytest -vv tests/test_file.py::test_name``
5. Use ``pytest --pdb`` for interactive debugging
6. Check :doc:`contributing` for reporting guidelines

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. Profile to identify bottleneck: ``pytest tests/test_performance_benchmarks.py -v``
2. Understand the hot path (CPU or GPU?)
3. Implement optimization
4. Benchmark improvement: ``pytest tests/test_performance_benchmarks.py -v``
5. Verify no regression: ``pytest --cov=glplot``
6. Document the change

Frequently Asked Questions
--------------------------

**Q: I found a bug, how do I report it?**

A: See the "Reporting Bugs" section in :doc:`contributing`. Include reproduction steps, environment details, and expected vs. actual behavior.

**Q: I want to add a new visualization type. Where do I start?**

A: Read the "Adding a New Renderer" section in :doc:`architecture`, then follow examples in ``glplot/renderers/``.

**Q: How do I test GPU code without a GPU?**

A: Use the OSMesa backend for headless testing. See "Headless Testing" in :doc:`testing`.

**Q: What's the performance target?**

A: 60 FPS for interactive exploration with millions of primitives. See "Performance Considerations" in :doc:`architecture`.

**Q: How should I structure my tests?**

A: Follow the Arrange-Act-Assert pattern described in :doc:`testing`. Each test should be independent and focused on one behavior.

Contributing Guidelines Summary
--------------------------------

- **Before coding**: Check :doc:`contributing` for setup and style guidelines
- **While coding**: Write tests alongside features, keep changes focused
- **Before PR**: Run formatter, full test suite, verify coverage
- **During review**: Respond constructively to feedback, make requested changes
- **After merge**: Celebrate! Your contribution improves GLPlot

For detailed guidelines, see :doc:`contributing`.

Getting Help
------------

- **Questions**: Open a GitHub issue with ``question`` label
- **Design discussions**: Use GitHub Discussions
- **Technical help**: Check documentation and existing issues first
- **Email**: Contact maintainers at lombardi@fhi-berlin.mpg.de
- **Discord/Slack**: Not available yet, but planned for future

Related Documentation
---------------------

- :doc:`../guide/installation` - Installation instructions
- :doc:`../guide/quickstart` - Quick start examples
- :doc:`../guide/basic-plotting` - Basic plotting reference
- :doc:`../api/index` - API documentation

Citation and Credits
--------------------

If you use GLPlot in your research, please cite:

::

    @software{glplot2026,
      title={GLPlot: High-Performance GPU-Accelerated Plotting Library for Python},
      author={Lombardi, Juan Manuel},
      year={2026},
      url={https://github.com/AkarisDimitry/GLPlot}
    }

See `CITATION.cff <../../CITATION.cff>`_ for more formats.
