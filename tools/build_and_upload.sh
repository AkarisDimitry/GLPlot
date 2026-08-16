#!/bin/bash
# GLPlot PyPI Upload Script
#
# Run from the repo root: tools/build_and_upload.sh

set -e
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

echo "🚀 GLPlot v$VERSION - PyPI Publication Workflow"
echo "============================================="
echo ""

# Check Python version
echo "✓ Checking Python environment..."
python_version=$(python3 --version | cut -d' ' -f2)
echo "  Python: $python_version"
echo ""

# Clean previous builds
echo "✓ Cleaning previous builds..."
rm -rf build dist *.egg-info
echo "  Removed old dist/ and build/ directories"
echo ""

# Check dependencies
echo "✓ Checking required packages..."
python3 -c "import build; import twine" || {
    echo "  ⚠️  Installing build tools..."
    python3 -m pip install build twine
}
echo "  Build tools ready"
echo ""

# Verify version consistency
echo "✓ Verifying version consistency..."
version_glplot=$(python3 -c "import glplot; print(glplot.__version__)")

echo "  pyproject.toml: $VERSION"
echo "  glplot.__version__: $version_glplot"

if [ "$VERSION" != "$version_glplot" ]; then
    echo "  ⚠️  WARNING: Version mismatch!"
fi
echo ""

# Build distribution
echo "✓ Building distribution packages..."
python3 -m build
echo "  Generated: dist/glplot-*.whl"
echo "  Generated: dist/glplot-*.tar.gz"
echo ""

# Validate packages
echo "✓ Validating packages..."
python3 -m twine check dist/*
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
echo "   python3 -m twine upload --repository testpypi dist/*"
echo "   python3 -m pip install --index-url https://test.pypi.org/simple/ glplot==$VERSION"
echo ""
echo "2. UPLOAD TO PRODUCTION PYPI:"
echo "   python3 -m twine upload dist/*"
echo ""
echo "3. VERIFY INSTALLATION:"
echo "   python3 -m pip install glplot==$VERSION"
echo "   python3 -c \"import glplot; print(glplot.__version__)\""
echo ""
echo "✅ Build complete and ready for upload!"
