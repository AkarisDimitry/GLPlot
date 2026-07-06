# Contributing to GLPlot

Thank you for your interest in contributing to GLPlot! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [issue list](https://github.com/AkarisDimitry/GLPlot/issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue
* **Describe the exact steps which reproduce the problem** with as many details as possible
* **Provide specific examples to demonstrate the steps** (include code or data)
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior
* **Explain which behavior you expected to see instead and why**
* **Include screenshots or animated GIFs if possible**
* **Include your environment details**: Python version, operating system, GPU model, relevant library versions

### Suggesting Enhancements

When creating enhancement suggestions, please include:

* **Use a clear and descriptive title** for the suggestion
* **Provide a step-by-step description of the suggested enhancement** with as many details as possible
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior** and **the proposed behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the Python code style guidelines (see below)
* Include appropriate test cases
* Update documentation to reflect your changes
* End all files with a newline

## Development Setup

### Prerequisites

* Python 3.9 or later
* pip or conda
* Git

### Setting Up Your Environment

```bash
git clone https://github.com/AkarisDimitry/GLPlot.git
cd GLPlot
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

For coverage report:
```bash
pytest --cov=glplot --cov-report=html
```

### Code Style

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
* Use meaningful variable and function names
* Limit lines to 100 characters where reasonable
* Use type hints for function signatures
* Write docstrings for public functions and classes

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Documentation

* Update relevant documentation when changing code
* Add docstrings to new functions and classes
* Include examples for new public APIs
* Update the README.md if your changes affect user-facing functionality

## Recognition

Contributors will be recognized in the project's documentation and repository metadata. Major contributors may be added to the project's authors list in `pyproject.toml`.

## Questions?

Feel free to open an issue with the `question` label or contact the maintainers at [lombardi@fhi-berlin.mpg.de](mailto:lombardi@fhi-berlin.mpg.de).

Thank you for contributing! 🎉
