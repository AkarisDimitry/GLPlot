GLPlot Architecture
===================

This document provides a technical overview of GLPlot's architecture, design patterns, and key subsystems. It's intended for developers contributing to or extending GLPlot.

High-Level Overview
--------------------

GLPlot is a GPU-accelerated plotting library providing Matplotlib-compatible Python APIs with an OpenGL-based rendering backend. The architecture decouples:

1. **User-Facing API** (pyplot): Familiar Matplotlib-style interface
2. **Scene Graph**: Layer-based representation of visual elements
3. **Rendering Engine**: GPU operations and state management
4. **Platform Integration**: GLFW windowing and OpenGL context

Key Design Principles
~~~~~~~~~~~~~~~~~~~~~

- **Reactive Rendering**: Only re-render when data or view changes
- **Layer Abstraction**: Decouple data representation from rendering
- **GPU-First**: Leverage GPU for heavy computations (line expansion, density)
- **Precision Management**: Handle extreme zooming via viewport-relative projections
- **Modular Managers**: Separate concerns (HUD, picking, effects, etc.)

Architecture Diagram
--------------------

::

    ┌─────────────────────────────────────────────────────────────┐
    │  User Application Layer                                      │
    │  glplot.pyplot (Matplotlib-compatible API)                  │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Scene Management Layer                                      │
    │  ├─ SceneData: Layer collections                            │
    │  ├─ BaseLayer & Subclasses: Data containers                │
    │  └─ LayerStyle & LayerDirtyState: Metadata & invalidation  │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Engine Core (GPULinePlot)                                   │
    │  ├─ State Management: Camera, Interaction, Cache            │
    │  ├─ Policy Manager: LOD & rendering mode decisions          │
    │  └─ Renderer Manager: Dispatch to appropriate renderers     │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  GPU Rendering Pipeline                                      │
    │  ├─ Renderers: Exact, Density, Interaction                 │
    │  ├─ Managers: HUD, Effects, Picking, Axis                  │
    │  └─ Framebuffers: Scene FBO, Effects FBO, Picking FBO      │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Platform Layer                                              │
    │  ├─ GLFW: Window management & events                        │
    │  ├─ OpenGL: GPU command submission                          │
    │  └─ Shaders: GLSL vertex/fragment programs                 │
    └─────────────────────────────────────────────────────────────┘

Core Components
---------------

1. Scene Management
~~~~~~~~~~~~~~~~~~~

**SceneData** (``core/legacy.py``)
    Centralized container for all visual elements:

    - ``lines``: Line families (y = ax + b)
    - ``strips``: Continuous polylines (connected points)
    - ``scatters``: Point clouds
    - ``texts``: Text annotations
    - ``layers``: Modern layer abstraction (replaces legacy lists)

    The scene maintains both legacy dataset lists (for backward compatibility) and the modern layer list.

**Layer Abstraction** (``core/layers.py``)
    All drawable elements inherit from ``BaseLayer``:

    - ``LineFamilyLayer``: Millions of analytical lines
    - ``PolylineLayer``: Connected line segments
    - ``ScatterLayer``: Point clouds with size/color mapping
    - ``PatchLayer``: Filled polygons and shapes
    - ``Layer3D``: 3D geometric primitives
    - ``TextLayer``: Text annotations

    Each layer has:

    - ``LayerStyle``: Visual properties (color, alpha, line width, etc.)
    - ``LayerDirtyState``: Fine-grained invalidation flags
    - ``bounds_world``: Cached bounding box

2. Rendering Engine
~~~~~~~~~~~~~~~~~~~

**GPULinePlot** (``engine.py``)
    Main engine class orchestrating the rendering loop:

    - Maintains scene, camera, and interaction state
    - Implements reactive event-based rendering loop
    - Dispatches to specialized renderers
    - Manages framebuffer lifecycle
    - Handles GLFW window and event dispatch

    Key state:

    - ``scene``: SceneData with all layers
    - ``camera``: CameraState (position, zoom, etc.)
    - ``cache``: CacheState (impostor caching)
    - ``frame``: FrameState (frame timing)
    - ``window``: GLFW window handle

    Rendering lifecycle:

    1. Check dirty flags → if clean, sleep until event
    2. Update camera and layer state
    3. Render to framebuffers (exact, density, picking)
    4. Apply post-processing effects
    5. Composite to screen

**RenderPolicyManager** (``policy.py``)
    Determines rendering mode and LOD:

    - Analyzes scene complexity
    - Selects mode: EXACT, DENSITY, or QUICK
    - Recommends subsampling for performance

    Decision factors:

    - Number of primitives
    - Display resolution
    - GPU fill-rate budget
    - User preferences (display_density flag)

**RendererManager** (``managers/renderer_manager.py``)
    Coordinates renderer dispatch:

    - Routes layers to appropriate renderers
    - Manages renderer state between frames
    - Caches compiled GPU resources

3. Rendering Subsystem
~~~~~~~~~~~~~~~~~~~~~~

**ExactLineRenderer** (``renderers/exact.py``)
    Renders geometric primitives with high precision:

    - Line families with analytical expansion in vertex shader
    - Continuous polylines with proper line joins
    - Implements viewport-relative center projection for precision
    - Uses GL_CLIP_DISTANCE for efficient viewport clipping
    - Supports variable line width per primitive

    Key innovations:

    - Y-clipping algorithm prevents coordinate overflow
    - Viewport-relative projection handles 10^7x zoom levels
    - Instanced rendering for millions of lines

**DensityRenderer** (``renderers/density.py``)
    Visualizes data density via accumulation:

    - Renders primitives additively to R32F framebuffer
    - Applies logarithmic normalization
    - Supports multiple density schemes (Linear, Log, Sqrt, etc.)
    - Includes HDR bloom effect for high-density regions

    Density pipeline:

    1. Render geometry to accumulation FBO (R32F)
    2. Extract luminance to temporary texture
    3. Blur luminance horizontally (ping-pong FBO)
    4. Blur luminance vertically (result in pong FBO)
    5. Composite with Reinhard tone mapping

**InteractionRenderer** (``renderers/interaction.py``)
    Enables interactive features:

    - Pixel-perfect picking (color-coded layer IDs)
    - Hover effects and selection highlighting
    - Affine reprojection for impostor caching

4. Manager Subsystems
~~~~~~~~~~~~~~~~~~~~~

**HudManager** (``managers/hud.py``)
    Renders on-screen interface:

    - Axes and tick labels
    - Legend
    - FPS counter
    - Color bars and colormaps
    - Uses ImGui for interactive UI elements

**PickingManager** (``managers/picking.py``)
    Efficient object selection:

    - Color-coded layer rendering to picking FBO
    - O(1) layer identification from mouse position
    - Returns layer ID and geometry info

**EffectManager** (``managers/effects.py``)
    Post-processing visual effects:

    - Bloom (glow effect)
    - Motion blur (optional)
    - Compositing and tone mapping

**AxisManager** (``managers/axis.py``)
    Axis and coordinate system handling:

    - Automatic tick generation
    - Label formatting
    - Supports linear and log scales

**CameraController** (``controllers.py``)
    Interactive camera control:

    - Pan, zoom, rotate
    - Mouse and keyboard input
    - Animation and smooth transitions

5. Shader System
~~~~~~~~~~~~~~~~

**GLSL Shaders** (``utils/shaders.py``)
    Low-level GPU programs:

    - Vertex shaders: Transform, clip, expand geometry
    - Fragment shaders: Color, lighting, density accumulation
    - Compute shaders: Future optimization opportunities

    Key vertex shader features:

    - Instanced rendering for data amplification
    - Analytical line expansion (y = ax + b → screen space)
    - Viewport-relative projection
    - Clip distance computation

    Key fragment shader features:

    - Proper alpha blending
    - Density accumulation
    - HDR tone mapping
    - Colormap lookup

6. Utility Modules
~~~~~~~~~~~~~~~~~~

**ExportManager** (``utils/export.py``)
    Save/print functionality:

    - PNG rasterization via OSMesa
    - Vector output (PDF, SVG) via Matplotlib
    - High-DPI rendering

**ShaderManager** (``utils/shaders.py``)
    Centralized shader compilation and caching

**MPL Bridge** (``utils/mpl_bridge.py``)
    Matplotlib integration:

    - Convert between GLPlot and Matplotlib objects
    - Apply Matplotlib colormaps

**GL Utils** (``utils/gl_utils.py``)
    OpenGL convenience functions:

    - Error checking
    - Capability detection

Rendering Pipeline
-------------------

Frame Lifecycle
~~~~~~~~~~~~~~~

Each frame follows this sequence:

1. **Event Handling** (Reactive Gate)

   - Poll GLFW for events (or sleep if no changes)
   - Update camera from input
   - Check layer dirty flags

2. **Upload Phase**

   - Copy modified layer data to GPU
   - Update uniforms (camera transform, etc.)
   - Rebuild VAOs if geometry changed

3. **Rendering Pass**

   Primary rendering pass (based on mode):

   - **EXACT Mode**: Direct geometry rendering
   - **DENSITY Mode**: Accumulation to R32F, then tone mapping
   - **QUICK Mode**: Subsampled/simplified geometry

4. **HUD Rendering**

   - Render ImGui interface
   - Overlay axes and legends
   - Display status information

5. **Post-Processing**

   - Apply bloom effect (if enabled)
   - Tone mapping and compositing
   - Copy result to screen

6. **Buffer Swap**

   - Present rendered frame to window

Dirty Flag System
~~~~~~~~~~~~~~~~~

Layers use fine-grained dirty flags to minimize GPU work:

.. code-block:: python

    class LayerDirtyState:
        data_dirty: bool      # Geometry/coordinates changed
        style_dirty: bool     # Colors, alpha, line width changed
        gpu_dirty: bool       # Needs GPU buffer update
        bounds_dirty: bool    # Bounding box invalid

When a layer property changes:

1. Update the property (e.g., layer.color = new_color)
2. Set appropriate dirty flag (style_dirty = True)
3. On next frame, manager detects flag and updates only affected GPU resources
4. After upload, flag is cleared

This minimizes redundant GPU transfers and re-compilation.

Impostor Caching
~~~~~~~~~~~~~~~~

For interactive pan/zoom without full re-render:

1. Render full scene to oversized cache texture (125-140% viewport)
2. On camera change, check if cache still covers viewport
3. If yes: draw screen-space quad mapped back to cache texture (O(1))
4. If no: re-render cache at new position

This maintains 60 FPS during smooth interactive exploration.

GPU Acceleration Strategy
--------------------------

Key GPU Optimizations
~~~~~~~~~~~~~~~~~~~~~

1. **Instanced Rendering**

   Instead of N draw calls for N primitives, use single instanced call:

   - Attributes: (a, b) coefficients for lines
   - Vertex shader expands to screen-space geometry
   - Reduces CPU-GPU synchronization overhead

2. **Analytical Geometry Expansion**

   Lines defined mathematically (y = ax + b) rather than tessellated:

   - GPU expands infinite line to viewport-clipped segment
   - Saves memory (4 floats vs millions of tessellated vertices)
   - Enables efficient line width implementation

3. **Viewport-Relative Projection**

   Prevents precision loss at extreme zoom levels:

   - CPU computes viewport center in double precision
   - Vertices transformed relative to center
   - Maintains float32 precision even at 10^7x zoom

4. **Hardware Clip Planes** (GL_CLIP_DISTANCE)

   Efficient viewport clipping without per-pixel tests:

   - Vertex shader computes distances to clip planes
   - GPU interpolates and discards before fragment shader
   - Avoids expensive discard() calls in fragment shader

5. **Framebuffer Arrangements**

   Specialized FBOs for different purposes:

   - **Scene FBO** (RGB8 or better): Main rendering
   - **Accumulation FBO** (R32F): Density accumulation
   - **Picking FBO** (R32U): Layer ID encoding
   - **Bloom FBOs** (Ping/Pong): Blur effects

GLFW Integration
----------------

Window and Event Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLFW provides:

- Native windowing and context management
- Event input (keyboard, mouse)
- Multi-monitor support
- Platform abstraction

GLPlot wraps GLFW in:

- ``GPULinePlot.window``: GLFW window handle
- ``CameraController``: Maps input events to camera state
- ``Interactive event loop``: poll/wait based on dirty flags

Event Flow
~~~~~~~~~~

::

    GLFW Event → CameraController → CameraState → dirty flag
       ↓
    Next frame_update() detects dirty → Re-render
       ↓
    Framebuffer swapped → User sees result

Reactive Event Loop
~~~~~~~~~~~~~~~~~~~

Unlike game engines, GLPlot sleeps when idle:

.. code-block:: python

    while not should_close:
        if scene_dirty or camera_dirty:
            render_frame()
            glfw.swap_buffers()
            dirty = False
        else:
            glfw.wait_events()  # Sleep until event

This reduces CPU and GPU power consumption during static analysis.

Key Module Descriptions
------------------------

``glplot/``
~~~~~~~~~~~

Main package. Public API:

- ``pyplot``: Matplotlib-style interface
- ``engine``: GPULinePlot class
- ``options``: Configuration dataclasses

``glplot/core/``
~~~~~~~~~~~~~~~~

Core abstractions:

- ``layers.py``: Layer definitions (LineFamilyLayer, etc.)
- ``context.py``: RenderContext for GPU state
- ``legacy.py``: Backward-compatible data structures

``glplot/renderers/``
~~~~~~~~~~~~~~~~~~~~~

Specialized rendering engines:

- ``exact.py``: High-precision geometry rendering
- ``density.py``: Density visualization
- ``axis.py``: Axis rendering
- ``patch.py``: Filled polygon rendering
- Specialized renderers: ``line_family.py``, ``scatter.py``, ``text.py``, ``geometry3d.py``

``glplot/managers/``
~~~~~~~~~~~~~~~~~~~~

Specialized subsystems:

- ``renderer_manager.py``: Dispatch and caching
- ``hud.py``: ImGui-based interface
- ``picking.py``: Selection and highlighting
- ``effects.py``: Post-processing
- ``axis.py``: Coordinate system management
- ``hud_state.py``: HUD configuration state

``glplot/utils/``
~~~~~~~~~~~~~~~~~

Utilities:

- ``shaders.py``: All shader sources and compilation
- ``export.py``: Save/print functionality
- ``mpl_bridge.py``: Matplotlib integration
- ``gl_utils.py``: OpenGL helpers
- ``preview.py``: Matplotlib preview fallback

``tests/``
~~~~~~~~~~

Comprehensive test suite:

- Unit tests: Layer, engine, API
- Integration tests: pyplot compliance
- Regression tests: Known issues
- Performance benchmarks: Throughput validation

Adding New Features
-------------------

Adding a New Renderer
~~~~~~~~~~~~~~~~~~~~~

To add rendering for a new primitive type:

1. Create ``glplot/renderers/new_type.py``
2. Inherit from renderer base (or implement interface)
3. Implement ``render(layer, context)`` method
4. Add shaders to ``glplot/utils/shaders.py``
5. Register in ``RendererManager.get_renderer()``
6. Add layer type to ``glplot/core/layers.py``
7. Add pyplot convenience function (if needed)
8. Write tests in ``tests/test_new_type.py``

Adding a New Manager
~~~~~~~~~~~~~~~~~~~~

For new functionality (effects, interactions, etc.):

1. Create ``glplot/managers/new_manager.py``
2. Inherit from manager base
3. Implement ``update(engine)`` and ``render()`` methods
4. Instantiate in ``GPULinePlot.__init__()``
5. Call methods from render loop
6. Add configuration to ``LayerStyle`` or ``EngineOptions``

Adding Camera Features
~~~~~~~~~~~~~~~~~~~~~~

1. Add camera state in ``CameraState`` (``core/legacy.py``)
2. Add controller logic in ``CameraController`` (``controllers.py``)
3. Add uniforms to shaders needing the state
4. Add pyplot convenience function
5. Test with interactive visual inspection

Performance Considerations
--------------------------

Profiling
~~~~~~~~~

Tools for identifying bottlenecks:

- ``cProfile``: CPU profiling
- ``py-spy``: Statistical profiling
- GPUView (Windows) or similar: GPU usage
- Benchmark suite: ``tests/test_performance_benchmarks.py``

Common Bottlenecks
~~~~~~~~~~~~~~~~~~

1. **CPU-GPU synchronization**: Minimize glFinish() calls
2. **Data transfers**: Batch uploads, use persistent mapped buffers
3. **Shader complexity**: Move work to faster stages or compute shaders
4. **Framebuffer copies**: Plan attachment layout to avoid readbacks
5. **Memory fragmentation**: Use consistent buffer allocation patterns

Optimization Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Profile before optimizing
- Prefer GPU solutions for data-parallel operations
- Batch similar operations (renderers, buffers)
- Use appropriate precision (float32 vs float64)
- Consider cache locality in shader code

Testing the Architecture
------------------------

Unit Testing
~~~~~~~~~~~~

Test individual components in isolation:

.. code-block:: bash

    pytest tests/test_layers.py -v

Integration Testing
~~~~~~~~~~~~~~~~~~~

Test subsystem interactions:

.. code-block:: bash

    pytest tests/test_pyplot_integration.py -v

Performance Testing
~~~~~~~~~~~~~~~~~~~

Benchmark critical paths:

.. code-block:: bash

    pytest tests/test_performance_benchmarks.py -v

Manual Visual Testing
~~~~~~~~~~~~~~~~~~~~~

Interactive validation:

.. code-block:: python

    import glplot.pyplot as plt
    # ... create plot ...
    plt.show()

Future Architecture Improvements
--------------------------------

Planned enhancements:

1. **Compute Shaders**: GPU-side data processing
2. **Indirect Rendering**: Reduce CPU overhead further
3. **GPU Memory Management**: Custom allocators for better performance
4. **Asynchronous Readback**: Non-blocking picking and export
5. **Multi-Window Support**: Shared GPU resources across windows
6. **Plugin System**: Third-party renderers and effects

Related Documentation
---------------------

- :doc:`testing` - Comprehensive testing guide
- :doc:`../guide/basic-plotting` - API usage examples
- `GLPlot Mathematical Formulation <../../GLPlot_Architecture_and_Mathematical_Formulation.md>`_
