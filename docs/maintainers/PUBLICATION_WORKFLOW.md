# GLPlot Publication Workflow v0.1.3

Complete checklist and instructions for publishing GLPlot to PyPI and setting up ReadTheDocs.

**Status**: ✅ Ready for publication  
**Version**: v0.1.3  
**Release Date**: July 7, 2026

---

## 📋 Pre-Publication Checklist

- [x] Type hints (80+ functions fully typed)
- [x] API documentation (Sphinx + ReadTheDocs)
- [x] Test suite (508 tests passing)
- [x] Code quality (Black, isort, flake8)
- [x] CI/CD pipeline (GitHub Actions)
- [x] License (MIT - OSI approved)
- [x] Git tag created (v0.1.3)
- [ ] GitHub Release created
- [ ] ReadTheDocs configured
- [ ] PyPI account ready
- [ ] Package uploaded to PyPI

---

## 🚀 Step 1: Create GitHub Release

GitHub Release is needed to mark this as an official version.

### Via GitHub Web Interface

1. Go to: https://github.com/AkarisDimitry/GLPlot/releases
2. Click "Draft a new release"
3. Fill in:
   - **Tag version**: v0.1.3 (already created locally ✅)
   - **Release title**: "GLPlot v0.1.3 - Production Ready Release"
   - **Description**: See release notes below

### Release Notes Content

```markdown
# GLPlot v0.1.3 - Production Ready Release

A comprehensive, high-performance GPU-accelerated plotting library for Python.

## 🎯 What's Included

### Core Features
- **High-Performance Plotting**: GPU-accelerated rendering at 60fps+ for millions of primitives
- **Matplotlib Compatible API**: Familiar interface for Python users
- **Comprehensive 2D/3D Visualization**: Lines, scatter, surfaces, bars, density plots, vector fields
- **Advanced Rendering**: Level-of-Detail (LOD), Screen-Space Ambient Occlusion (SSAO), custom colormaps

### Quality Improvements (This Release)
- ✅ **Complete Type Hints** (80+ functions): Full type coverage for internal and public APIs
- ✅ **Formal API Documentation**: Sphinx + ReadTheDocs integration ready
- ✅ **508 Tests Passing**: Comprehensive test suite with excellent coverage
- ✅ **Code Quality**: Black formatting, isort import ordering, full linting compliance
- ✅ **Cross-Platform Support**: Linux, macOS, Windows with verified CI/CD

## 📦 Installation

### PyPI
```bash
pip install glplot
```

### From Source
```bash
git clone https://github.com/AkarisDimitry/GLPlot.git
cd GLPlot
pip install -e .
```

## 📚 Documentation

- **API Docs**: [glplot.readthedocs.io](https://glplot.readthedocs.io/)
- **GitHub**: [AkarisDimitry/GLPlot](https://github.com/AkarisDimitry/GLPlot)
- **Examples**: See `examples/gallery/` for 19 complete plotting examples

## 🧪 Testing

All 508 tests pass on Python 3.9-3.12 (Linux, macOS, Windows).

```bash
pytest tests/
```

## 📄 Citation

```bibtex
@software{lombardi2026glplot,
  author = {Lombardi, Juan Manuel},
  title = {GLPlot: High-Performance GPU-Accelerated Plotting Library},
  year = {2026},
  url = {https://github.com/AkarisDimitry/GLPlot}
}
```

### Release Details

**Features**:
- 30+ plotting functions (plot, scatter, bar, hist, contour, etc.)
- 3D visualization (plot3d, scatter3d, bar3d, mesh3d, wireframe3d)
- Advanced options: SSAO, LOD, custom colormaps, image overlays

**Documentation**:
- Comprehensive API reference (auto-generated from docstrings)
- 7 user guides covering all features
- Developer documentation
- 250+ working code examples

**Quality**:
- Full type hints across codebase
- 508 passing tests
- GitHub Actions CI/CD
- Black + isort formatting

**Performance**:
- Handles 10M+ points at 60fps+
- Efficient GPU memory management
- Benchmark scripts included

---

**Python**: 3.9 - 3.12  
**License**: MIT (OSI-approved)  
**Repository**: https://github.com/AkarisDimitry/GLPlot
```

---

## 📖 Step 2: Configure ReadTheDocs

ReadTheDocs will auto-host the documentation online.

### Setup Instructions

1. **Sign in to ReadTheDocs**
   - Go to: https://readthedocs.io/
   - Sign in with GitHub account

2. **Import Project**
   - Click "Import a Project"
   - Select "Import Manually"
   - Fill in:
     - **Project name**: glplot
     - **Project URL**: https://github.com/AkarisDimitry/GLPlot
     - **Repository URL**: https://github.com/AkarisDimitry/GLPlot.git
     - **Repository type**: Git
     - **Default branch**: main

3. **Configure Settings**
   - **Admin** → **Settings**:
     - **Documentation type**: Sphinx (HTML)
     - **Python configuration file**: `docs/source/conf.py` (already set in `.readthedocs.yml`)
     - **Python version**: 3.11
     - **Programming language**: Python

4. **Advanced Settings** (already configured in `.readthedocs.yml`)
   - The repo includes `.readthedocs.yml` with:
     - Python 3.11 environment
     - Sphinx configuration from `docs/source/conf.py`
     - Auto-installation from `docs/requirements.txt`
     - Build formats: HTML, PDF, ePub

5. **Build Docs**
   - ReadTheDocs will automatically build when you push to main
   - First build status: https://readthedocs.org/projects/glplot/builds/
   - Live documentation: https://glplot.readthedocs.io/

### What Gets Built

- ✅ **API Reference**: Auto-generated from docstrings (5 modules)
- ✅ **User Guides**: 7 comprehensive tutorials
- ✅ **Developer Docs**: Contributing, architecture, testing guides
- ✅ **Gallery**: Example scripts organized by category
- ✅ **Search**: Full-text search across all docs
- ✅ **Versions**: Support for multiple Python versions

---

## 🐍 Step 3: Prepare for PyPI Publication

### 3a. Get PyPI Account

1. Go to: https://pypi.org/account/register/
2. Create account (or sign in if you have one)
3. Set up two-factor authentication (recommended)

### 3b. Create PyPI API Token

For secure automated uploads (recommended):

1. Go to: https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Fill in:
   - **Token name**: `glplot-upload` (or similar)
   - **Scope**: Entire account
4. Copy token (you won't see it again!)

### 3c. Configure Local Environment

Store credentials in `~/.pypirc` (only read by your user):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

**Permissions**: `chmod 600 ~/.pypirc`

### 3d. Test on Test PyPI First (Recommended)

```bash
# Create distribution files
python -m build

# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ glplot==0.1.3
```

### 3e. Upload to Production PyPI

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

### 3f. Verify Publication

```bash
# Install from PyPI
pip install glplot==0.1.3

# Test import
python -c "import glplot; print(glplot.__version__)"
```

---

## ✅ Verification Checklist

### After Publishing to PyPI

- [ ] Package appears on https://pypi.org/project/glplot/
- [ ] Installation works: `pip install glplot==0.1.3`
- [ ] Import works: `python -c "import glplot"`
- [ ] Version correct: Check `glplot.__version__ == "0.1.3"`
- [ ] Dependencies installed: `pip show glplot`

### After ReadTheDocs Setup

- [ ] Documentation builds successfully
- [ ] Live at: https://glplot.readthedocs.io/
- [ ] API reference auto-generated
- [ ] Search functionality works
- [ ] Mobile view responsive

### After GitHub Release

- [ ] Release visible at: https://github.com/AkarisDimitry/GLPlot/releases/v0.1.3
- [ ] Release notes display correctly
- [ ] Download links work

---

## 🔄 Post-Publication

### Update Documentation Link

Add to README.md:

```markdown
📚 **[Documentation](https://glplot.readthedocs.io/)** — Full API reference, guides, and examples
```

### Announce Release

- Update GitHub repo description
- Link documentation badge in README
- Consider announcing on:
  - Scientific Python forums
  - Reddit r/learnprogramming or r/Python
  - Python mailing lists

### Future Maintenance

- Automatic documentation updates on each push to `main`
- Auto-rebuild on new releases
- Monitor ReadTheDocs build status
- Handle any dependency updates promptly

---

## 📝 Files Ready for Publication

**Configuration**:
- ✅ `pyproject.toml` — v0.1.3, all metadata configured
- ✅ `.readthedocs.yml` — RTD configuration
- ✅ `CITATION.cff` — Citation metadata
- ✅ `LICENSE` — MIT license
- ✅ `CHANGELOG.md` — Version history

**Documentation**:
- ✅ `README.md` — Installation, examples, features
- ✅ `docs/` — Complete Sphinx documentation
- ✅ `paper.md` — Academic paper for SoftwareX

**Tests**:
- ✅ `tests/` — 508 passing tests
- ✅ `.github/workflows/` — CI/CD pipeline

**Type Hints**:
- ✅ 80+ functions fully typed
- ✅ All internal/public APIs covered

---

## 🎯 Summary

| Task | Status | Effort | Impact |
|------|--------|--------|--------|
| GitHub Release | ⏳ Manual | 5 min | Marks official version |
| ReadTheDocs Setup | ⏳ Manual | 5 min | Online documentation |
| PyPI Account | ⏳ Manual | 10 min | Public availability |
| PyPI Upload | ⏳ Manual | 2 min | `pip install glplot` works |

**Total time to publication**: ~20-30 minutes of manual steps

---

## 📞 Troubleshooting

### PyPI Upload Fails

```bash
# Check token is valid
python -m twine --version

# Upload with verbose output
python -m twine upload --verbose dist/*
```

### ReadTheDocs Build Fails

- Check build logs: https://readthedocs.org/projects/glplot/builds/
- Verify `.readthedocs.yml` is at repo root
- Ensure `docs/requirements.txt` has all dependencies
- Test locally: `cd docs && make html`

### PyPI Package Not Found

- Give PyPI 1-2 minutes to sync
- Check: https://pypi.org/project/glplot/
- Verify version number in `pyproject.toml`

---

**Ready to go live! 🚀**

Generated: July 7, 2026
