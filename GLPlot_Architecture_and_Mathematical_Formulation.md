# GLPlot: High-Performance GPU-Accelerated Scientific Plotting Engine
## Software Engineering Design, Mathematical Formulation, and Core Innovations

This document serves as the technical specification and mathematical reference for the **GLPlot** visualization engine. The structure adheres to the manuscript guidelines of the Elsevier academic journal *SoftwareX*, providing the software engineering details and mathematical rigor (in LaTeX format) required for high-impact scientific software publications.

---

## Software Metadata

| Field | Description |
| :--- | :--- |
| **Software Name** | GLPlot (Graphics Library Plot) |
| **Developers** | Core Visualization & Engineering Team |
| **Contact** | dimitry@glplot.org |
| **Year Released** | 2026 |
| **Software Version** | 0.1.3 |
| **Programming Language** | Python (3.9+), GLSL (Core Profile 330) |
| **Key Dependencies** | PyOpenGL >= 3.1.6, GLFW >= 2.5, NumPy >= 1.23, SciPy >= 1.9, Matplotlib >= 3.6, ImGui >= 2.0 |
| **Software License** | MIT License |
| **Code Repository** | [https://github.com/AkarisDimitry/GLPlot](https://github.com/AkarisDimitry/GLPlot) |

---

## 1. Motivation and Scientific Impact

Interactive visualization of large-scale datasets is crucial in computational physics, bioinformatics, civil engineering, and data science. However, traditional CPU-bound plotting libraries (such as Matplotlib) fail to maintain smooth interactive frame rates (panning, rotation, zooming) at 60 frames per second (FPS) once the number of geometric elements exceeds $10^5$. 

Existing GPU-accelerated alternatives in the Python ecosystem exhibit notable limitations:
1. **Steep Learning Curve**: Low-level libraries (e.g., VisPy or raw OpenGL bindings) require scientists to manually manage VAOs, VBOs, and shader programs.
2. **GPU Precision Limitations**: Extreme zooming (sub-micron scales) causes single-precision floating-point (32-bit float) GPU computations to collapse due to catastrophic cancellation.
3. **Absence of Density Accumulation**: Plotting millions of overlapping continuous curves under standard alpha blending quickly saturates color channels, obliterating the underlying statistical distribution.

GLPlot addresses these challenges by providing an environment **fully compatible with Matplotlib's familiar API (`glplot.pyplot`)**, driven by a high-performance C/OpenGL backend. Its core innovations reside in GPU-side analytical geometry expansion, fill-rate budget optimization, and high-dynamic-range (HDR) tone-mapping applied to scientific data analysis.

---

## 2. Software Architecture and Execution Pipeline

GLPlot adopts a modular design centered around a **Decoupled Scene Graph** and a **Reactive Event-Gated Rendering Loop**.

```mermaid
graph TD
    subgraph Frontend API (User Space)
        Pyplot[glplot.pyplot] -->|Define Layers & Styles| Scene[SceneData & Layers]
    end

    subgraph Core Engine (Control Plane)
        Engine[GPULinePlot] -->|Render Request| Policy[RenderPolicyManager]
        Policy -->|LOD & Mode Decision| RM[RendererManager]
    end

    subgraph Post-Processing Pipeline (Post-FX)
        RM -->|Pass 1: Geometry to FBO| SceneFBO[Scene Framebuffer]
        SceneFBO -->|Pass 2: Luma Extraction| LumaFBO[Extract Framebuffer]
        LumaFBO -->|Pass 3: Horizontal Blur| PingFBO[Ping Framebuffer]
        PingFBO -->|Pass 4: Vertical Blur| PongFBO[Pong Framebuffer]
        PongFBO -->|Pass 5: Reinhard Mapping| DefaultFBO[Default Screen Framebuffer]
    end

    subgraph GPU Subsystems
        RM -->|Exact Draw Pass| Renderers[Primitive Renderers]
        RM -->|O(1) Pixel Sampling| Picking[PickingManager]
        RM -->|Pixel Warping| Impostor[InteractionRenderer]
    end
```

### 2.1 Frame Lifecycle and Reactive Event Gating
Unlike gaming engines that render continuously in a busy loop, GLPlot operates reactively. If the camera transform and layers remain unchanged (the engine's *dirty flags* are false), the frame loop halts redraws and sleeps the execution thread via `glfw.wait_events()`. This reduces idle CPU and GPU utilization to nearly $0\%$, making it ideal for scientific environments where plots remain static during analysis.

---

## 3. Core Technical Innovations and Mathematical Formulations

### 3.1 Analytical Line-Family Shader Expansion & Y-Clipping
When studying chaotic systems or thermodynamic phases, it is common to plot families of infinite straight lines defined by $y = a_i x + b_i$ over a global domain $x \in [x_{min}, x_{max}]$. Generating segments for millions of such lines in CPU memory is highly inefficient. 

GLPlot handles this by transferring only the coefficient matrix $[a_i, b_i]$ to the GPU as **instanced attributes**. The vertex shader (`WIDE_LINES_INSTANCED_VS`) analytically computes the visible segment of each line relative to the active viewport:
$$\mathbf{v} = [x_{left}, x_{right}, y_{bottom}, y_{top}]^T$$

The primary horizontal clipping boundaries are defined as:
$$x_A = \max(x_{left}, x_{min}), \quad x_B = \min(x_{right}, x_{max})$$

#### The Y-Clipping Algorithm
For steep slopes ($a_i \gg 1$), the calculated endpoints $y = a_i x + b_i$ can yield extremely large values (e.g., $10^9$), causing coordinate overflow in the clip-space stages of the GPU. GLPlot solves this by performing analytical vertical clipping over an extended tolerance zone $\epsilon = 0.1 \times (y_{top} - y_{bottom})$:

For endpoint $A$:
* If $y_A > y_{top} + \epsilon$ and $a_i \neq 0$:
  $$x_A \leftarrow x_A + \frac{(y_{top} + \epsilon) - y_A}{a_i}, \quad y_A \leftarrow y_{top} + \epsilon$$
* If $y_A < y_{bottom} - \epsilon$ and $a_i \neq 0$:
  $$x_A \leftarrow x_A + \frac{(y_{bottom} - \epsilon) - y_A}{a_i}, \quad y_A \leftarrow y_{bottom} - \epsilon$$

*(The same clipping logic is applied to endpoint $B$).*

If $x_A \ge x_B$ after clipping, or if the entire line falls outside the vertical bounds ($y_A, y_B > y_{top} + \epsilon$ or $y_A, y_B < y_{bottom} - \epsilon$), the instance is discarded by moving its vertex coordinates outside the normalized device coordinate (NDC) space:
$$\mathbf{x}_{ndc} = [2.0, 2.0, 2.0, 1.0]^T$$

---

### 3.2 Viewport-Relative Center Projection
To bypass catastrophic cancellation in 32-bit floats during high-zoom scale operations (e.g., zooming to sub-micron scales on large offsets):
1. The engine computes the viewport center on the CPU in double precision (64-bit float):
   $$c_x = \frac{x_{left} + x_{right}}{2}, \quad c_y = \frac{y_{bottom} + y_{top}}{2}$$
2. The center $\mathbf{c} = [c_x, c_y]^T$ is passed as a uniform.
3. The shader translates coordinates relative to this center prior to scaling to NDC space:
   $$\mathbf{p}_{rel} = \mathbf{p}_{world} - \mathbf{c} + \mathbf{t}_{layer}$$
   $$\mathbf{p}_{ndc} = \mathbf{p}_{rel} \odot \mathbf{s}_{ndc}$$
   where $\mathbf{s}_{ndc} = \left[ \frac{2}{x_{right} - x_{left}}, \frac{2}{y_{top} - y_{bottom}} \right]^T$ represents the NDC scaling factor.

This limits active coordinates to the stable precision range of float32 ($[-1.0, 1.0]$), preventing rendering artifacts like line jittering or clipping during deep zooms up to $10^{-7}$.

---

### 3.3 Hardware-Based Clipping Optimizations (`GL_CLIP_DISTANCE`)
When visualizing sub-regions of a dataset, discarding out-of-viewport fragments via `discard` statements in fragment shaders causes severe fill-rate bottlenecks. This occurs because the GPU must execute the fragment shader stages before rejecting the pixel, bypassing early depth-testing optimizations (*Early-Z*).

GLPlot utilizes **Hardware-Based Clip Planes**. The vertex shader writes the distances to the viewport boundaries:
$$gl\_ClipDistance[0] = w_x - x_{min}$$
$$gl\_ClipDistance[1] = x_{max} - w_x$$
$$gl\_ClipDistance[2] = w_y - y_{min}$$
$$gl\_ClipDistance[3] = y_{max} - w_y$$

The GPU linearly interpolates these values across primitives. If any interpolated distance in a fragment is negative, the hardware **automatically halts rasterization for that fragment** before executing the fragment shader, minimizing fill-rate overhead.

---

### 3.4 R32F Accumulation with HDR Tone Mapping (Reinhard)
Visualizing overlapping line families using standard alpha blending leads to rapid color saturation. GLPlot addresses this via a two-stage accumulation process:

#### Stage 1: Scientific Density Accumulation
Fragments are rendered additively into an offscreen framebuffer with 32-bit single-channel floating-point precision (`GL_R32F`):
$$D(x, y) = \sum_{i=1}^{M} \alpha_i \cdot \delta_i(x, y)$$

#### Stage 2: Logarithmic Normalization & HDR Compression
To capture density variations spanning several orders of magnitude, the accumulated value $D(x,y)$ is processed in the compositing fragment shader using a logarithmic transfer function:
$$n = \frac{\log_{10}(1.0 + D(x, y) \cdot g)}{\max\left(\log_{10}(1.0 + D_{max} \cdot g), \,\, 10^{-6}\right)}$$

If Bloom (glow effect) is active, the combined intensities of the scene and blurred texture can exceed unity ($C_{raw} = C_{scene} + I_{bloom} \cdot C_{blur} > 1.0$). Rather than clamping values to $1.0$ (which leads to flat, blown-out regions), GLPlot applies the **Reinhard HDR Tone Mapping formula**:
$$\mathbf{C}_{final} = \frac{\mathbf{C}_{raw}}{\mathbf{C}_{raw} + \mathbf{1}}$$

This function maps the range $[0, \infty)$ asymptotically to $[0, 1)^3$, preserving color gradients in high-density regions.

---

### 3.5 Impostor Caching and Affine Reprojection
To maintain 60 FPS during interactive pan/zoom operations, GLPlot renders scene data into an offscreen cache texture that covers an expanded viewport (padded by $25\%$ to $40\%$).

When the user pans or zooms, the engine draws a single screen-space quad and maps coordinates back to the cached texture using inverse affine transformations:
$$u_{cache} = \frac{\left(x_l + u \cdot (x_r - x_l)\right) - x_{c,l}}{x_{c,r} - x_{c,l}}$$
$$v_{cache} = \frac{\left(y_b + v \cdot (y_t - y_b)\right) - y_{c,b}}{y_{c,t} - y_{c,b}}$$

This updates the screen instantly by warping the cached image, postponing geometry re-renders until interactive inputs stop.

---

### 3.6 Fill-Rate Complexity Budgeting (Width-Aware LOD)
Traditional Level-of-Detail (LOD) subsampling relies solely on primitive count. However, a line with a thickness of 20 pixels consumes 20 times the fill-rate (rasterized pixels) of a 1-pixel line.

GLPlot introduces a **Fill-Rate-Based LOD Engine**:
1. The projected pixel footprint is estimated as:
   $$F_{total} = N_{lines} \cdot W_{viewport} \cdot w_{px} + \sum_{j} L_{j, px} \cdot w_{j, px}$$
   where the screen length of polyline $L_{j, px}$ is evaluated using a fast strided vertex estimator with step size $S$:
   $$L_{j, px} = S \cdot \sum_{k=1}^{M/S} \left\| \mathbf{x}_{k \cdot S, px} - \mathbf{x}_{(k-1) \cdot S, px} \right\|$$
2. A fill-rate budget is set based on the active viewport area:
   $$F_{target} = c \cdot W_{viewport} \cdot H_{viewport}$$
3. The primitive keep probability is computed:
   $$P_{keep} = \min\left(1.0, \,\, \frac{F_{target}}{F_{total}}\right)$$
4. The value of $P_{keep}$ is passed to the vertex shader. Each instanced thread hashes its instance ID:
   ```glsl
   uint h = uint(gl_InstanceID);
   h ^= h >> 17; h *= 0xed5ad4bbu; h ^= h >> 11;
   h *= 0xac4c1b51u; h ^= h >> 15; h *= 0x31848babu;
   float rnd = float(h & 0x00FFFFFFu) * (1.0 / 16777215.0);
   ```
   If $rnd > P_{keep}$, the vertex shader shifts the coordinates to infinity, skipping rasterization stages.

---

### 3.7 Height-Based Analytical Ambient Occlusion (SSAO 3D)
Generating screen-space ambient occlusion (SSAO) for 3D meshes or bar charts usually requires multi-pass depth and normal buffers, which can limit performance in Python.

GLPlot approximates ambient occlusion analytically in a single pass based on the normalized height of the vertex $z_n \in [0, 1]$:
$$\text{cavity} = 1.0 - \text{smoothstep}(0.10, \,\, 0.95, \,\, z_n)$$
$$\text{rim} = \text{smoothstep}(0.00, \,\, 0.18, \,\, z_n)$$
$$\text{AO}(z_n) = \text{clamp}(1.0 - s \cdot \text{cavity} \cdot \text{rim}, \,\, 0.18, \,\, 1.0)$$

This factor modulates the diffuse material reflection. Valleys and bases are darkened to simulate contact shadows, while peaks retain full reflectivity, enhancing depth perception with zero runtime overhead.

---

### 3.8 GPU-Based Constant Time $O(1)$ Interactive Picking
Identifying which line or object lies under the cursor in a scene containing millions of primitives is computationally expensive on the CPU.

GLPlot implements a deferred integer ID picking pass:
1. During the picking pass (rendered to a single-channel integer FBO `GL_R32I` with blending and multisampling disabled), each element writes its unique instance identifier:
   $$\text{ID} = \text{ID}_{offset} + gl\_InstanceID + 1$$
2. Upon user interaction, the engine reads the single pixel under the cursor $(x, y)$:
   $$\text{ID}_{read} = \text{glReadPixels}(x, \,\, H - y, \,\, 1, \,\, 1, \,\, \text{GL\_RED\_INTEGER}, \,\, \text{GL\_INT})$$
3. The returned ID is decoded to determine the parent layer and element index in constant $O(1)$ time.

---

## 4. API Capabilities (`glplot.pyplot`)

GLPlot replicates the Matplotlib API structure, providing GPU-accelerated drawing functions:

### 4.1 2D Plotting Capabilities
* **`plot(*args, **kwargs)`**: Renders continuous polylines optimized for large datasets. Supports custom widths, colors, and line styles.
* **`scatter(x, y, s=None, c=None, cmap=None, **kwargs)`**: Renders point clouds. Uses SDF-based point rasterization to draw antialiased circles with custom outlines.
* **`plot_lines(a, b, x_range=None, **kwargs)`**: Draws infinite line families directly on the GPU from slope and intercept coefficients.
* **`bar(x, height, width=0.8, bottom=None, **kwargs)`**: Generates 2D bar charts.
* **`hist(x, bins=10, density=False, **kwargs)`**: Computes bin counts on the CPU and generates the matching 2D bar mesh on the GPU.
* **`hist2d(x, y, bins=100, cmap="magma", **kwargs)`**: Generates 2D density maps.
* **`imshow(X, extent=None, origin="upper", cmap="viridis", **kwargs)`**: Displays 2D matrices using textured quads with bilinear filtering.
* **`pcolormesh(X, Y, C, cmap="viridis", **kwargs)`**: Renders quad meshes using indexed geometry patches.
* **`contour(X, Y, Z, levels=10, cmap="viridis", **kwargs)`**: Computes and draws 2D contour lines.
* **`contourf(X, Y, Z, levels=10, cmap="viridis", **kwargs)`**: Computes and draws filled contour regions.
* **`arrow(x, y, dx, dy, **kwargs)` and `quiver(x, y, u, v, **kwargs)`**: Renders 2D vector fields with analytical arrowheads.
* **`annotate(text, xy, xytext=None, **kwargs)`**: Renders screen-aligned text labels using a dynamic bitmap font atlas.

### 4.2 3D Plotting Capabilities
* **`plot3d(x, y, z, **kwargs)`**: Draws continuous 3D polylines.
* **`scatter3d(x, y, z, c=None, cmap=None, **kwargs)`**: Renders 3D point clouds with perspective-scaled point sizes.
* **`plot_surface(X, Y, Z, cmap="viridis", **kwargs)`**: Draws 3D surfaces with analytical height-based ambient occlusion.
* **`plot_wireframe(X, Y, Z, **kwargs)`**: Renders structured 3D wireframes.
* **`mesh3d(vertices, faces, **kwargs)`**: Renders general triangular 3D meshes.
* **`volume3d(x, y, z, values, threshold=None, **kwargs)`**: Visualizes scalar fields as point-cloud isosurfaces.
* **`quiver3d(x, y, z, u, v, w, **kwargs)`**: Renders 3D vector fields.

#### Hexagonal vs. Box Geometry in `bar3d`
* **`bar3d(x, y, z, dx, dy, dz, shape="box", gap=0.0, **kwargs)`**: Renders 3D bar graphs using two user-selected geometries:
  1. **`shape="box"`**: Traditional rectangular boxes (6 faces, 12 triangles).
  2. **`shape="hex"`**: **Hexagonal prisms** (8 faces, 20 triangles). This shape is designed to visualize compact hexagonal grid layouts, reducing spatial overlap.
  
  Both options support a `gap` factor to shrink the bases, defining clear boundaries between adjacent columns. They integrate with the SSAO height engine to enhance visual depth.

---

## 5. Post-Processing Pipeline Architecture (Bloom)

GLPlot includes a post-processing system managed by `EffectManager` that implements a two-pass separable blur:

1. **Luminance Threshold Extraction Pass**: The fragment shader (`BLOOM_EXTRACT_FS`) extracts pixels that exceed a threshold value $T_{glow}$:
   $$C_{extract} = \begin{cases} C_{scene} & \text{if } 0.2126 R + 0.7152 G + 0.0722 B > T_{glow} \\ (0, 0, 0, 1) & \text{otherwise} \end{cases}$$
2. **Two-Pass Separable Gaussian Blur**:
   Rather than performing a 2D neighborhood lookup of complexity $O(D^2)$, GLPlot splits the Gaussian kernel into horizontal and vertical passes executed on ping-pong framebuffers (`ping_fbo` and `pong_fbo`):
   * **Horizontal Pass**:
     $$C_H(x, y) = \sum_{i=-4}^{4} w_i \cdot C_{extract}\left(x + i \cdot r \cdot \Delta x, \,\, y\right)$$
   * **Vertical Pass**:
     $$C_V(x, y) = \sum_{i=-4}^{4} w_i \cdot C_H\left(x, \,\, y + i \cdot r \cdot \Delta y\right)$$
   where the 1D symmetric weights are defined as:
   $$\mathbf{w} = [0.227027, \,\, 0.1945946, \,\, 0.1216216, \,\, 0.054054, \,\, 0.016216]^T$$
   This reduces complexity to $O(2D)$ per fragment, allowing large blur radii to render at interactive frame rates.

---

## 6. Conclusions and Impact

GLPlot provides a solution for interactive rendering of large-scale datasets within the Python scientific ecosystem. By combining a Matplotlib-compatible API with a GPU-instanced rendering pipeline, it enables real-time exploration of large datasets.

The integration of analytical ambient occlusion, fill-rate-based LOD, Reinhard HDR tone mapping, and $O(1)$ GPU picking makes this engine suitable for data analysis and visualization tasks in engineering and scientific applications.
