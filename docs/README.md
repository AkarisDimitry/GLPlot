# GLPlot Documentation

Complete API documentation built with Sphinx and hosted on ReadTheDocs.

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML

Generate HTML documentation:

```bash
make html
```

Open in browser:

```bash
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Build PDF

Generate PDF documentation (requires LaTeX):

```bash
make pdf
```

Output: `_build/pdf/GLPlot.pdf`

### Build ePub

Generate ePub eBook:

```bash
make epub
```

### Clean Build

Remove all build artifacts:

```bash
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main index page
│   ├── guide/               # User guides
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   ├── basic-plotting.rst
│   │   ├── 2d-plotting.rst
│   │   ├── 3d-visualization.rst
│   │   ├── advanced-features.rst
│   │   └── performance-tips.rst
│   ├── api/                 # API reference (auto-generated)
│   │   ├── core.rst
│   │   ├── plotting.rst
│   │   ├── layers.rst
│   │   ├── managers.rst
│   │   └── utilities.rst
│   ├── dev/                 # Developer guides
│   │   ├── contributing.rst
│   │   ├── architecture.rst
│   │   └── testing.rst
│   ├── gallery/
│   │   └── gallery.rst      # Examples and gallery
│   └── _static/
│       └── custom.css       # Custom styling
├── Makefile                 # Build automation
└── requirements.txt         # Python dependencies
```

## ReadTheDocs Integration

### Setup (One-Time)

1. Go to [readthedocs.io](https://readthedocs.io/)
2. Sign in with GitHub account
3. Import repository
4. Select `main` branch
5. Build will start automatically
6. Documentation will be published at: `https://glplot.readthedocs.io/`

### Automatic Builds

- On every push to `main` → docs rebuild automatically
- PR previews available for documentation changes
- Builds take ~1-2 minutes

### Configuration

ReadTheDocs reads from:
- `.readthedocs.yml` — Build settings, Python version, dependencies
- `docs/conf.py` — Sphinx settings
- `docs/requirements.txt` — Python packages needed

## Writing Documentation

### Adding New Guides

1. Create `.rst` file in `docs/source/guide/`
2. Add reference to `docs/source/index.rst` toctree
3. Build locally to verify: `make html`
4. Commit and push

### Improving API Docs

API docs are auto-generated from docstrings:

1. Improve docstrings in `glplot/` source code
2. Use Google-style format (configured via Napoleon)
3. Include examples in docstrings
4. Rebuild: `make html`

Example docstring:

```python
def plot(x, y, fmt=None, **kwargs):
    """Plot lines or markers and return the line objects.
    
    This function creates a line plot with given data and optional
    formatting.
    
    Args:
        x (array-like): X coordinates. Shape (N,).
        y (array-like): Y coordinates. Shape (N,).
        fmt (str, optional): Format string (e.g., 'r--', 'b-o').
            Defaults to None.
        **kwargs: Additional keyword arguments passed to plot styling.
    
    Returns:
        list: List of Layer objects added to plot.
    
    Examples:
        Plot a sine wave:
        
        >>> import glplot as gplt
        >>> import numpy as np
        >>> x = np.linspace(0, 2*np.pi, 100)
        >>> y = np.sin(x)
        >>> gplt.plot(x, y, 'b-', label='sin(x)')
        >>> gplt.show()
    """
```

### RST Formatting Tips

- Headings use `====` for H1, `----` for H2, `^^^^` for H3
- Code blocks start with `.. code-block:: python`
- Cross-references: `` :doc:`path/to/file` `` or `` :ref:`label` ``
- Inline code: ` `backticks` `
- External links: `` `text <url>`_ ``

## Sphinx Extensions Used

- **autodoc**: Auto-generate API docs from docstrings
- **autosummary**: Summaries for functions/classes
- **napoleon**: Parse Google-style docstrings
- **intersphinx**: Link to NumPy/Matplotlib docs
- **viewcode**: Show source code links
- **mathjax**: Render math equations

## Troubleshooting

### Build Fails: "Module not found"

Make sure `glplot` package is installed:

```bash
pip install -e .
```

### Autodoc Doesn't Show Members

Check `conf.py` `autodoc_default_options`:
- `members: True` — Include member functions/classes
- `undoc-members: True` — Include undocumented members
- `show-inheritance: True` — Show parent classes

### Sphinx Warnings

Run with verbose output to see warnings:

```bash
make clean
make html SPHINXOPTS="-v"
```

### ReadTheDocs Build Fails

Check:
1. `.readthedocs.yml` Python version matches `pyproject.toml`
2. All imports in `conf.py` are available in `requirements.txt`
3. No syntax errors in RST files (run `sphinx-build -n` for nitpicky mode)

## Publishing

### Manual Push to ReadTheDocs

Changes auto-build when you push to GitHub. To manually trigger:

1. Visit [readthedocs.io dashboard](https://readthedocs.io/dashboard/)
2. Select GLPlot project
3. Click "Build"

### Versions

ReadTheDocs automatically builds:
- `latest` — Most recent commit on main branch
- `stable` — Most recent tagged release
- Pull request previews (for documentation branches)

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [ReadTheDocs Guide](https://docs.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Napoleon: Google/NumPy Style Guide](https://sphinx-napoleon.readthedocs.io/)

## Quick Checklist

- [ ] Created account at readthedocs.io
- [ ] Imported GLPlot repository
- [ ] Verified `.readthedocs.yml` in repo root
- [ ] First build succeeded
- [ ] Documentation available at glplot.readthedocs.io
- [ ] Added link in README.md or GitHub repo description
