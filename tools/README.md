# GLPlot Development Tools

This directory contains diagnostic and utility scripts for GLPlot development and troubleshooting.

## Tools Overview

### `check_gl_capabilities.py`
**Purpose**: Verify OpenGL capabilities on your system

**What it checks**:
- OpenGL version and vendor
- Point size limits and granularity
- Framebuffer and texture support
- Required extensions

**Usage**:
```bash
python tools/check_gl_capabilities.py
```

**When to use**:
- Troubleshooting rendering issues
- Verifying system compatibility
- Environment setup validation

---

### `validate_runtime_math.py`
**Purpose**: Validate camera controller mathematical operations

**What it checks**:
- Asymmetric bounds handling
- Screen-to-world coordinate conversion
- Various window aspect ratios
- Large offset precision

**Usage**:
```bash
python tools/validate_runtime_math.py
```

**When to use**:
- Debugging camera transformation issues
- Verifying coordinate math accuracy
- Testing edge cases in camera behavior

---

### `diagnose_camera_state.py`
**Purpose**: Diagnose camera state persistence and viewport behavior

**What it checks**:
- Initial camera state
- View limit persistence
- Pan and zoom operations
- Window resize effects
- Bounds computation

**Usage**:
```bash
python tools/diagnose_camera_state.py
```

**When to use**:
- Debugging camera state issues
- Verifying viewport transformations
- Testing interactive camera behavior

---

### `build_and_upload.sh`
**Purpose**: Build the sdist/wheel, validate them with `twine check`, and print the
`twine upload` commands for TestPyPI/PyPI

**What it checks**:
- `pyproject.toml`'s `version` matches `glplot.__version__` (warns on mismatch)

**Usage**:
```bash
tools/build_and_upload.sh
```

**When to use**:
- Preparing a release

---

## Common Issues and Solutions

### Issue: "GL_PROGRAM_POINT_SIZE is DISABLED"

**Symptom**: Point markers not rendering or have fixed size

**Solution**:
1. Update GPU drivers to latest version
2. Check if OpenGL version is 3.2+ (modern profiles required)
3. Verify graphics card supports GL_PROGRAM_POINT_SIZE

### Issue: Math validation fails

**Symptom**: Camera transformations incorrect or precision issues

**Solution**:
1. Run `validate_runtime_math.py` to identify which operation fails
2. Check camera controller implementation
3. Verify floating-point precision is adequate

### Issue: Camera state not persisting

**Symptom**: View resets when panning/zooming

**Solution**:
1. Run `diagnose_camera_state.py` to trace state changes
2. Check if `set_view()` is being called unexpectedly
3. Verify camera state is properly stored

---

## Quick Diagnostics Checklist

```bash
# 1. Check OpenGL capabilities
python tools/check_gl_capabilities.py

# 2. Validate math operations
python tools/validate_runtime_math.py

# 3. Check camera state behavior
python tools/diagnose_camera_state.py
```

If all three pass, your environment should be ready for GLPlot development.

---

## Adding New Tools

When creating new diagnostic tools:

1. Place in `tools/` directory
2. Make executable: `chmod +x tools/my_tool.py`
3. Add shebang: `#!/usr/bin/env python3`
4. Add docstring at module level
5. Use clear output formatting (sections with `---`)
6. Document in this README

Example template:

```python
#!/usr/bin/env python3
"""Brief description of what this tool does.

Longer description of purpose and use cases.

Usage:
    python tools/my_tool.py
"""

import sys


def main():
    """Main function."""
    print("=" * 70)
    print("Tool Title")
    print("=" * 70)
    
    # ... implementation ...
    
    print("=" * 70)
    print("✓ Complete")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

---

## Support

For issues with these tools:
1. Run tool with `-v` or `--verbose` flag (if supported)
2. Check output for specific error messages
3. Report issues on GitHub with tool output attached

---

These three scripts are the whole directory — see `check_gl_capabilities.py`,
`validate_runtime_math.py`, and `diagnose_camera_state.py` above for what each does.
