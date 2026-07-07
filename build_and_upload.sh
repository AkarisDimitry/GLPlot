#!/bin/bash
# GLPlot v0.1.3 PyPI Upload Script

set -e

echo "🚀 GLPlot v0.1.3 - PyPI Publication Workflow"
echo "============================================="
echo ""

# Check Python version
echo "✓ Checking Python environment..."
python_version=$(python --version | cut -d' ' -f2)
echo "  Python: $python_version"
echo ""

# Clean previous builds
echo "✓ Cleaning previous builds..."
rm -rf build dist *.egg-info
echo "  Removed old dist/ and build/ directories"
echo ""

# Check dependencies
echo "✓ Checking required packages..."
python -c "import build; import twine" || {
    echo "  ⚠️  Installing build tools..."
    pip install build twine
}
echo "  Build tools ready"
echo ""

# Verify version consistency
echo "✓ Verifying version consistency..."
version_pyproject=$(python -c "import tomllib; f=open('pyproject.toml','rb'); print([v for v in tomllib.load(f)['project'].values() if isinstance(v, str) and v.startswith('0.')][0])" 2>/dev/null || echo "")
version_glplot=$(python -c "import glplot; print(glplot.__version__)")

echo "  pyproject.toml: $version_pyproject"
echo "  glplot.__version__: $version_glplot"

if [ "$version_pyproject" != "$version_glplot" ]; then
    echo "  ⚠️  WARNING: Version mismatch!"
fi
echo ""

# Build distribution
echo "✓ Building distribution packages..."
python -m build
echo "  Generated: dist/glplot-*.whl"
echo "  Generated: dist/glplot-*.tar.gz"
echo ""

# Validate packages
echo "✓ Validating packages..."
python -m twine check dist/*
echo "  Packages are valid"
echo ""

# Show what was built
echo "✓ Build Summary"
echo "  Distribution files created:"
ls -lh dist/
echo ""

# Instructions
echo "📋 Next Steps:"
echo ""
echo "1. TEST ON TEST PYPI (Recommended):"
echo "   python -m twine upload --repository testpypi dist/*"
echo "   pip install --index-url https://test.pypi.org/simple/ glplot==0.1.3"
echo ""
echo "2. UPLOAD TO PRODUCTION PYPI:"
echo "   python -m twine upload dist/*"
echo ""
echo "3. VERIFY INSTALLATION:"
echo "   pip install glplot==0.1.3"
echo "   python -c \"import glplot; print(glplot.__version__)\""
echo ""
echo "✅ Build complete and ready for upload!"
