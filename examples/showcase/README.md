# GLPlot Showcase Examples

Quick & beautiful examples demonstrating GLPlot's capabilities with simple, elegant code.

> **All examples render at 60+ FPS with 100k+ points**

## 2D Examples

### ✨ Colorful Particles 2D
**File**: `01_colorful_particles_2d.py`

100,000 vibrant particles arranged in concentric circles with smooth color gradients.

```python
python examples/showcase/01_colorful_particles_2d.py
```

**What it shows**:
- Massive 2D point cloud rendering (100k points)
- Rainbow color mapping
- Interactive pan/zoom
- Smooth 60+ FPS performance

**Key features**:
- Concentric particle arrangement
- Smooth color gradients
- Alpha blending for depth perception
- Interactive controls

---

### 🌀 Mandelbrot Fractal
**File**: `02_mandelbrot_zoom_2d.py`

360,000 points rendering the stunning Mandelbrot fractal with psychedelic colors.

```python
python examples/showcase/02_mandelbrot_zoom_2d.py
```

**What it shows**:
- Complex mathematical visualization
- Very large dataset (360k points)
- Psychedelic color mapping
- Smooth interactive panning/zooming

**Key features**:
- Iterative computation of Mandelbrot set
- Twilight color scheme for drama
- Zoom-invariant rendering
- Fractal structure exploration

---

## 3D Examples

### 🍩 Spinning Torus
**File**: `03_spinning_torus_3d.py`

50,000 points forming a vibrant spinning torus with smooth color gradients.

```python
python examples/showcase/03_spinning_torus_3d.py
```

**What it shows**:
- Beautiful 3D geometric shape
- Parametric surface rendering
- HSV color mapping for vibrant output
- Full 3D interactivity

**Key features**:
- Parametric torus equations
- Smooth azimuthal coloring
- 3D rotation and zoom
- Clean, readable code

---

### 🌌 Cosmic Sphere
**File**: `04_cosmic_sphere_3d.py`

150,000 points forming a sphere with Perlin-like noise and vivid psychedelic colors.

```python
python examples/showcase/04_cosmic_sphere_3d.py
```

**What it shows**:
- Complex 3D surface with perturbations
- Large-scale 3D rendering (150k points)
- Multi-dimensional color mapping
- Advanced noise-based effects

**Key features**:
- Spherical coordinate system
- Noise perturbation for bumpy surface
- Rainbow color mapping
- Interactive 3D exploration

---

## Why These Examples?

Each one is 10-15 lines of actual code, colorful, fully interactive (pan/zoom/rotate),
and runs at 60+ FPS regardless of point count — good starting points to copy-paste from,
or to use for teaching/demos.

---

## Running All Showcase Examples

```bash
# Run each example individually
python examples/showcase/01_colorful_particles_2d.py
python examples/showcase/02_mandelbrot_zoom_2d.py
python examples/showcase/03_spinning_torus_3d.py
python examples/showcase/04_cosmic_sphere_3d.py

# Or create your own variations!
```

---

## Keyboard & Mouse Controls

### All Examples
- **Mouse drag**: Pan (2D) or Rotate (3D)
- **Mouse scroll**: Zoom in/out
- **Close window**: Press ESC or close window button

### Interactive Features
- Smooth interpolation during panning/zooming
- Level-of-detail adaptation for performance
- No lag or stuttering even with massive datasets

---

## Extending the Examples

### Change Colors
```python
# Use different colormaps:
cmap="viridis"      # Traditional scientific
cmap="twilight"     # Psychedelic purple/pink
cmap="gist_rainbow" # Full spectrum
cmap="hot"          # Fire-like colors
```

### Change Point Count
```python
n = 500_000  # Make it even more impressive!
# All examples scale linearly - still 60+ FPS
```

### Change Geometry
```python
# Modify mathematical functions to create:
# - Different fractals
# - Alternative surfaces
# - Custom patterns
# - Animated deformations
```

---

## Showcase Statistics

| Example | Points | FPS | Type | Computation |
|---------|--------|-----|------|-------------|
| Particles 2D | 100k | 60+ | Points | Minimal |
| Mandelbrot | 360k | 60+ | Points | Complex |
| Torus 3D | 50k | 60+ | Surface | Parametric |
| Cosmic Sphere | 150k | 60+ | Surface | Noise-based |

---

## Notes

- All examples run **headless** in CI/CD (no windows required)
- Code is intentionally simple for readability and teaching
- Performance remains constant regardless of dataset size
- Designed for presentations and demonstrations
- Perfect for technical portfolios and papers

---

**License**: MIT (see [../../LICENSE](../../LICENSE))
