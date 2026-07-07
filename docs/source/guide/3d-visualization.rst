3D Visualization
================

GLPlot provides powerful native 3D visualization capabilities with GPU-accelerated rendering.
All 3D functions support interactive camera control and can handle millions of geometric primitives.
This guide covers the main 3D plotting functions and best practices for working with volumetric data.

3D Line Plots
-------------

Basic 3D Line Plots
^^^^^^^^^^^^^^^^^^^

Plot a single or multiple 3D curves using ``plot3d``. The function connects points with a line in 3D space:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Generate a 3D spiral
   t = np.linspace(0, 4*np.pi, 1000)
   x = np.cos(t)
   y = np.sin(t)
   z = t / (4*np.pi)

   gplt.figure("3D Spiral", figsize=(8, 6))
   gplt.plot3d(x, y, z, 'b-', linewidth=1.5, label='Spiral')
   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.zlabel('Z')
   gplt.title('3D Spiral Curve')
   gplt.legend()
   gplt.show()

Parametric 3D Curves
^^^^^^^^^^^^^^^^^^^^

Use parametric equations to create complex 3D shapes:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Trefoil knot
   t = np.linspace(0, 4*np.pi, 2000)
   r = np.sin(3*t)
   x = r * np.cos(2*t)
   y = r * np.sin(2*t)
   z = np.cos(3*t)

   gplt.figure("Trefoil Knot", figsize=(8, 6))
   gplt.plot3d(x, y, z, color='#FF6B6B', linewidth=1.2)
   gplt.title('Trefoil Knot')
   gplt.show()

Camera Parameters
^^^^^^^^^^^^^^^^^

Control the viewing angle using elevation and azimuth:

.. code-block:: python

   gplt.figure("Camera Views", figsize=(8, 6))
   t = np.linspace(0, 6*np.pi, 1500)
   x = np.cos(t)
   y = np.sin(t)
   z = t / (6*np.pi)

   gplt.plot3d(x, y, z, 'g-', linewidth=2,
               elev=45,      # Elevation angle (degrees)
               azim=-30,     # Azimuth angle (degrees)
               scale_z=0.8)  # Z-axis scale factor
   gplt.title('Spiral with Custom View')
   gplt.show()

**Camera Control Parameters:**

- **elev** (float): Camera elevation angle in degrees. Higher values look down from above (0-90). Default is 30.
- **azim** (float): Camera azimuth angle in degrees. Rotation around vertical axis (-180 to 180). Default is -60.
- **scale_z** (float): Z-axis scale factor to adjust aspect ratio. Default is 0.7.

Multiple 3D Lines
^^^^^^^^^^^^^^^^^

Plot multiple curves with different styles:

.. code-block:: python

   gplt.figure("Multiple 3D Curves", figsize=(10, 7))

   t = np.linspace(0, 4*np.pi, 500)

   gplt.plot3d(t, np.sin(t), np.cos(t), 'r-', linewidth=1.5, label='sin/cos')
   gplt.plot3d(t, np.cos(t), np.sin(t), 'b--', linewidth=1.5, label='cos/sin')
   gplt.plot3d(t, t/(2*np.pi), np.zeros_like(t), 'g:', linewidth=2, label='linear')

   gplt.legend()
   gplt.title('Multiple 3D Parametric Curves')
   gplt.show()

3D Scatter Plots (Point Clouds)
-------------------------------

Basic 3D Scatter Plot
^^^^^^^^^^^^^^^^^^^^^

Create a scatter plot in 3D space using ``scatter3d``:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(42)
   n = 10000

   # Random point cloud
   x = rng.normal(size=n)
   y = rng.normal(size=n)
   z = rng.normal(size=n)

   gplt.figure("3D Point Cloud", figsize=(8, 6))
   gplt.scatter3d(x, y, z, s=2, alpha=0.6)
   gplt.title('Random 3D Point Cloud')
   gplt.show()

Colormapped Point Clouds
^^^^^^^^^^^^^^^^^^^^^^^^

Color points based on a scalar field:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(42)
   n = 50000

   t = rng.uniform(0, 10*np.pi, n)
   r = 0.05 * t + rng.uniform(0, 1, n)
   x = r * np.cos(t)
   y = r * np.sin(t)
   z = 0.05 * t + rng.normal(scale=0.2, size=n)

   # Color by z-coordinate (height)
   gplt.figure("Spiral Point Cloud", figsize=(9, 6))
   gplt.scatter3d(x, y, z, c=z, cmap='turbo', s=1.5, alpha=0.8,
                  label='Spiral cloud')
   gplt.colorbar(label='Height (Z)')
   gplt.title('3D Spiral Point Cloud with Colormap')
   gplt.legend()
   gplt.show()

Large Point Cloud Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GLPlot efficiently handles massive point clouds through GPU acceleration:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(1)

   # 1 million points
   n = 1_000_000
   t = rng.uniform(0, 24*np.pi, n)
   radius = 0.08 * t + 0.7 * rng.random(n)
   x = radius * np.cos(t)
   y = radius * np.sin(t)
   z = 0.08 * t + rng.normal(scale=0.35, size=n)

   gplt.figure("Large Point Cloud", figsize=(10, 7))
   gplt.scatter3d(x, y, z, c=z, cmap='turbo', s=1.2, alpha=0.75,
                  elev=24, azim=-42, scale_z=0.85,
                  label='1M points')
   gplt.title('One Million Point 3D Cloud')
   gplt.legend()
   gplt.show()

**Performance Tips for Point Clouds:**

- Use smaller point sizes (s < 2) for large datasets
- Increase alpha transparency to reduce visual clutter
- Use appropriate color mapping to enhance structure
- Downsample very large clouds (>10M points) if interaction is needed

3D Surfaces and Wireframes
--------------------------

Basic Surface Plot
^^^^^^^^^^^^^^^^^^

Create a 3D surface from a 2D grid using ``plot_surface``:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create grid
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)
   X, Y = np.meshgrid(x, y)

   # Compute surface
   Z = np.sin(np.sqrt(X**2 + Y**2))

   gplt.figure("3D Surface", figsize=(9, 6))
   gplt.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
   gplt.title('Radial Wave Surface')
   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.zlabel('Z')
   gplt.show()

Surface with Colormap
^^^^^^^^^^^^^^^^^^^^^

Use colormaps to visualize Z-values:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   x = np.linspace(-3, 3, 150)
   y = np.linspace(-3, 3, 150)
   X, Y = np.meshgrid(x, y)

   # Complex surface
   r = np.hypot(X, Y) + 1e-6
   Z = 1.8 * np.sin(3.4 * r) / (1 + 0.22 * r**2) + \
       0.32 * np.cos(2.5 * X) * np.sin(2.0 * Y)

   gplt.figure("Complex Surface", figsize=(10, 7))
   gplt.plot_surface(X, Y, Z, cmap='turbo', elev=28, azim=-48,
                     scale_z=0.92, alpha=0.92, rstride=2, cstride=2,
                     label='Complex wave')
   gplt.title('Complex Surface with Turbo Colormap')
   gplt.legend()
   gplt.show()

Wireframe Meshes
^^^^^^^^^^^^^^^^

Visualize mesh structure using ``plot_wireframe``:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   x = np.linspace(-5, 5, 80)
   y = np.linspace(-5, 5, 80)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(X) * np.cos(Y)

   gplt.figure("Wireframe Mesh", figsize=(9, 6))
   gplt.plot_wireframe(X, Y, Z, color='navy', linewidth=0.5,
                       rstride=4, cstride=4)
   gplt.title('Wireframe Surface')
   gplt.show()

Combining Surface and Wireframe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overlay a wireframe on a solid surface:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   x = np.linspace(-3, 3, 100)
   y = np.linspace(-3, 3, 100)
   X, Y = np.meshgrid(x, y)

   r = np.hypot(X, Y) + 1e-6
   Z = 1.8 * np.sin(3.4 * r) / (1 + 0.22 * r**2)

   gplt.figure("Surface + Wireframe", figsize=(10, 7))

   # Solid surface
   gplt.plot_surface(X, Y, Z, cmap='turbo', alpha=0.8,
                     elev=28, azim=-48, scale_z=0.92,
                     rstride=2, cstride=2, label='Surface')

   # Wireframe overlay
   gplt.plot_wireframe(X, Y, Z + 0.01, color='black', linewidth=0.4,
                       rstride=12, cstride=12, alpha=0.5,
                       elev=28, azim=-48, scale_z=0.92)

   gplt.legend()
   gplt.title('Surface with Wireframe Overlay')
   gplt.show()

Surface Stride Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

Control mesh density with stride parameters:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   x = np.linspace(-5, 5, 200)
   y = np.linspace(-5, 5, 200)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(X) * np.cos(Y)

   gplt.figure("Surface Stride Control", figsize=(10, 7))

   # rstride: row decimation, cstride: column decimation
   # Higher values = fewer triangles = faster rendering
   gplt.plot_surface(X, Y, Z, cmap='plasma',
                     rstride=4,     # Draw every 4th row
                     cstride=4,     # Draw every 4th column
                     alpha=0.85, elev=25, azim=-50)

   gplt.title('Decimated Surface Mesh (stride=4)')
   gplt.show()

3D Bar Charts
-------------

Basic 3D Bar Chart
^^^^^^^^^^^^^^^^^^

Create bars at 3D positions using ``bar3d``:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create bar positions
   x = np.array([1, 2, 3, 4, 5])
   y = np.array([1, 1, 1, 1, 1])
   z = np.zeros_like(x)

   # Bar heights
   heights = np.array([1.5, 2.3, 1.8, 2.8, 1.2])

   gplt.figure("3D Bar Chart", figsize=(9, 6))
   gplt.bar3d(x, y, z, dx=0.4, dy=0.4, dz=heights,
              color='steelblue', alpha=0.8)
   gplt.title('Simple 3D Bar Chart')
   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.show()

Colormapped 3D Bars
^^^^^^^^^^^^^^^^^^^

Color bars based on values:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # 2D grid of bars
   x = np.repeat(np.arange(5), 5)
   y = np.tile(np.arange(5), 5)
   z = np.zeros_like(x)

   # Heights based on product of coordinates
   heights = (x + 1) * (y + 1)

   gplt.figure("Colormapped 3D Bars", figsize=(10, 7))
   gplt.bar3d(x, y, z, dx=0.35, dy=0.35, dz=heights,
              c=heights, cmap='viridis',
              elev=25, azim=-45, scale_z=0.8,
              label='Grid heights')
   gplt.colorbar(label='Bar Height')
   gplt.title('2D Grid of 3D Bars with Colormap')
   gplt.legend()
   gplt.show()

Hexagonal and Custom Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use different bar shapes for visual variety:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(42)
   n = 50

   x = rng.uniform(0, 10, n)
   y = rng.uniform(0, 10, n)
   z = np.zeros(n)
   heights = rng.uniform(1, 5, n)

   gplt.figure("Hexagonal Bars", figsize=(10, 7))
   gplt.bar3d(x, y, z, dx=0.3, dy=0.3, dz=heights,
              c=heights, cmap='turbo',
              shape='hex',  # Hexagonal bars
              alpha=0.85, elev=30, azim=-60)
   gplt.title('Hexagonal 3D Bar Chart')
   gplt.show()

Vector Fields in 3D (quiver3d)
------------------------------

Basic 3D Vector Field
^^^^^^^^^^^^^^^^^^^^^

Visualize a vector field in 3D using ``quiver3d``:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create grid
   grid = np.linspace(-2, 2, 11)
   x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')

   # Define vector field (e.g., radial field)
   r2 = x**2 + y**2 + z**2 + 0.1
   u = x / r2
   v = y / r2
   w = z / r2

   gplt.figure("3D Vector Field", figsize=(10, 7))
   gplt.quiver3d(x.ravel(), y.ravel(), z.ravel(),
                 u.ravel(), v.ravel(), w.ravel(),
                 scale=0.5, normalize=False,
                 color='steelblue', linewidth=0.8,
                 label='Radial field')
   gplt.scatter3d([0], [0], [0], color='red', s=30, label='origin')
   gplt.title('3D Radial Vector Field')
   gplt.legend()
   gplt.show()

Normalized Vector Fields
^^^^^^^^^^^^^^^^^^^^^^^^

Show direction without magnitude variation:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   grid = np.linspace(-1.5, 1.5, 8)
   x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')

   # Circular field
   u = -y
   v = x
   w = 0.5 * np.sin(z)

   gplt.figure("Normalized Vector Field", figsize=(10, 7))
   gplt.quiver3d(x.ravel(), y.ravel(), z.ravel(),
                 u.ravel(), v.ravel(), w.ravel(),
                 scale=0.6, normalize=True,
                 color='#FF6B6B', linewidth=0.9,
                 head_length=0.15, head_width=0.08)
   gplt.title('3D Circular Vector Field (Normalized)')
   gplt.show()

Complex Vector Fields
^^^^^^^^^^^^^^^^^^^^^

Combine vector fields with other 3D visualizations:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(42)

   # Large vector field lattice
   grid = np.linspace(-2.4, 2.4, 19)
   x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')

   # Lorenz-like field
   sigma, rho, beta = 5.4, 10.8, 1.75
   u = sigma * (y - x) + 0.45 * np.sin(2.2 * z)
   v = x * (rho - z * 2.6) - y + 0.55 * np.cos(1.9 * x)
   w = x * y - beta * z + 0.35 * np.sin(2.7 * y)

   speed = np.sqrt(u**2 + v**2 + w**2)
   mask = speed.ravel() > np.quantile(speed, 0.70)

   # Particle cloud following field
   cloud_n = 100000
   t = rng.uniform(0, 10*np.pi, cloud_n)
   radius = rng.gamma(2.0, 0.72, cloud_n)
   cloud_x = radius * np.cos(t)
   cloud_y = radius * np.sin(t)
   cloud_z = 0.3 * radius * np.sin(t * 0.7)
   cloud_energy = np.exp(-0.15 * radius**2)

   gplt.figure("Complex Vector Field", figsize=(12, 8))

   # Background volumetric data
   gplt.volume3d(cloud_x, cloud_y, cloud_z, cloud_energy,
                 threshold=0.16, cmap='magma', alpha=0.15, s=0.8,
                 elev=28, azim=-45, label='Flow samples')

   # Vector field overlay
   gplt.quiver3d(x.ravel()[mask], y.ravel()[mask], z.ravel()[mask],
                 u.ravel()[mask], v.ravel()[mask], w.ravel()[mask],
                 scale=0.3, normalize=True,
                 color=(0.15, 0.9, 1.0, 0.85),
                 linewidth=0.7, head_length=0.14,
                 elev=28, azim=-45, label='Vector field')

   gplt.title('Lorenz-like Vector Field over Volume')
   gplt.legend()
   gplt.show()

Vector Field Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fine-tune arrow appearance:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   grid = np.linspace(-1, 1, 7)
   x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')
   u, v, w = -y, x, 0.3*z

   gplt.figure("Custom Vector Field", figsize=(9, 6))
   gplt.quiver3d(x.ravel(), y.ravel(), z.ravel(),
                 u.ravel(), v.ravel(), w.ravel(),
                 scale=0.8,              # Arrow length scaling
                 normalize=True,          # Unit direction vectors
                 color='#4ECDC4',         # Teal color
                 linewidth=1.2,           # Arrow shaft thickness
                 head_length=0.2,         # Arrowhead length
                 head_width=0.1,          # Arrowhead width
                 elev=35, azim=-50)
   gplt.title('Customized 3D Vector Field')
   gplt.show()

Camera Control and Viewing Angles
---------------------------------

Setting Custom Viewing Angles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All 3D functions support elevation and azimuth parameters:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create sample data
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(np.sqrt(X**2 + Y**2))

   # View from different angles
   angles = [
       (20, -30, "Top view"),
       (60, 0, "Side view"),
       (10, 90, "Front view"),
       (45, 45, "Isometric view"),
   ]

   for elev, azim, title in angles:
       gplt.figure(title, figsize=(7, 5))
       gplt.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                        elev=elev, azim=azim)
       gplt.title(f'{title} (elev={elev}, azim={azim})')
       gplt.show()

Adjusting Z-Axis Scale
^^^^^^^^^^^^^^^^^^^^^^

Control aspect ratio with the scale_z parameter:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   t = np.linspace(0, 6*np.pi, 1000)
   x = np.cos(t)
   y = np.sin(t)
   z = t / (6*np.pi)

   scales = [0.5, 0.7, 1.0, 1.5]

   for scale in scales:
       gplt.figure(f"scale_z={scale}", figsize=(7, 5))
       gplt.plot3d(x, y, z, 'b-', linewidth=2,
                   scale_z=scale, elev=30, azim=-60)
       gplt.title(f'Spiral with Z-scale = {scale}')
       gplt.show()

Interactive Camera Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^

During interactive rendering, use keyboard shortcuts to control the camera:

- **Middle mouse drag** or **Scroll + drag**: Rotate view
- **Right mouse drag**: Pan view
- **Scroll wheel**: Zoom in/out
- **Home key** or **R key**: Reset view
- **H key**: Toggle HUD (Head-Up Display)

Mixing 2D and 3D in Same Scene
------------------------------

Combining 2D Axes with 3D Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can add 2D elements to a 3D figure:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   t = np.linspace(0, 4*np.pi, 500)
   x = np.cos(t)
   y = np.sin(t)
   z = t / (4*np.pi)

   gplt.figure("3D with 2D Projections", figsize=(10, 7))

   # 3D line
   gplt.plot3d(x, y, z, 'b-', linewidth=1.5, elev=30, azim=-60,
               label='3D curve')

   # 2D projection on XY plane
   gplt.plot(x, y, 'r--', alpha=0.5, linewidth=1,
             label='XY projection')

   gplt.legend()
   gplt.title('3D Curve with 2D Projections')
   gplt.show()

3D Scatter with 2D Overlay
^^^^^^^^^^^^^^^^^^^^^^^^^^

Overlay 2D contours with 3D data:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(42)

   # 3D point cloud
   n = 5000
   x = rng.normal(size=n)
   y = rng.normal(size=n)
   z = rng.normal(size=n)

   gplt.figure("3D Cloud with 2D Grid", figsize=(10, 7))

   # 3D scatter
   gplt.scatter3d(x, y, z, s=1.5, alpha=0.3, c='blue',
                  elev=25, azim=-45, label='Point cloud')

   # 2D grid on XY plane
   grid_x = np.linspace(-3, 3, 20)
   grid_y = np.linspace(-3, 3, 20)
   gplt.plot(grid_x, 0*grid_x, 'k-', alpha=0.2, linewidth=0.5)
   for gy in grid_y[::2]:
       gplt.plot(grid_x, gy*np.ones_like(grid_x), 'k-',
                 alpha=0.2, linewidth=0.5)

   gplt.title('3D Cloud with 2D Grid Reference')
   gplt.legend()
   gplt.show()

Performance Tips for Large 3D Datasets
--------------------------------------

Point Cloud Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

For very large point clouds (>1M points):

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Generate 10 million point cloud
   n = 10_000_000
   rng = np.random.default_rng(42)

   x = rng.normal(size=n)
   y = rng.normal(size=n)
   z = rng.normal(size=n)

   gplt.figure("Optimized Large Cloud", figsize=(10, 7))

   # Optimization strategies
   gplt.scatter3d(x, y, z,
                  s=0.8,         # Small point size
                  alpha=0.3,     # Transparent
                  c=z, cmap='turbo',
                  elev=20, azim=-40,
                  label='10M points')
   gplt.title('Large Point Cloud Rendering')
   gplt.show()

Mesh Decimation
^^^^^^^^^^^^^^^

Control mesh complexity for surfaces:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # High-resolution mesh
   x = np.linspace(-5, 5, 500)
   y = np.linspace(-5, 5, 500)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(X) * np.cos(Y)

   gplt.figure("Decimated Surface", figsize=(10, 7))

   # Use stride to reduce triangle count
   gplt.plot_surface(X, Y, Z, cmap='viridis',
                     rstride=8,        # Every 8th row
                     cstride=8,        # Every 8th column
                     alpha=0.85)

   gplt.title('250k triangles with stride=8')
   gplt.show()

Vector Field Sampling
^^^^^^^^^^^^^^^^^^^^^

Subsample dense vector fields:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create fine grid
   grid = np.linspace(-3, 3, 51)
   x_full, y_full, z_full = np.meshgrid(grid, grid, grid, indexing='ij')

   u = -y_full
   v = x_full
   w = 0.3 * np.sin(z_full)

   # Subsample every 3rd point
   step = 3
   mask = (np.arange(len(grid)) % step == 0).reshape(-1, 1, 1)
   x = x_full[mask].reshape(-1)
   y = y_full[mask].reshape(-1)
   z = z_full[mask].reshape(-1)
   u = u[mask].reshape(-1)
   v = v[mask].reshape(-1)
   w = w[mask].reshape(-1)

   gplt.figure("Sampled Vector Field", figsize=(9, 6))
   gplt.quiver3d(x, y, z, u, v, w, scale=0.6, normalize=True)
   gplt.title('Subsampled Vector Field')
   gplt.show()

Practical 3D Examples
---------------------

Example 1: Atmospheric Data Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize temperature field at different altitudes:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Synthetic atmospheric data
   rng = np.random.default_rng(42)
   lats = np.linspace(-90, 90, 30)
   lons = np.linspace(-180, 180, 30)
   alts = np.linspace(0, 10000, 15)

   LAT, LON, ALT = np.meshgrid(lats, lons, alts, indexing='ij')

   # Temperature decreases with altitude
   temperature = 288 - 0.0065 * ALT + 5 * np.sin(LAT/30) + \
                 3 * np.sin(LON/60) + rng.normal(scale=0.5, size=LAT.shape)

   # Select data above 5000m
   mask = ALT.ravel() >= 5000
   gplt.figure("Atmospheric Temperature", figsize=(11, 8))
   gplt.scatter3d(LAT.ravel()[mask], LON.ravel()[mask], ALT.ravel()[mask],
                  c=temperature.ravel()[mask],
                  cmap='coolwarm', s=2.5, alpha=0.6,
                  elev=20, azim=-30)
   gplt.colorbar(label='Temperature (K)')
   gplt.xlabel('Latitude')
   gplt.ylabel('Longitude')
   gplt.zlabel('Altitude (m)')
   gplt.title('3D Atmospheric Temperature Field')
   gplt.show()

Example 2: Molecular Structure Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize atoms and bonds in 3D:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Simple benzene-like ring structure
   n_atoms = 6
   angles = np.linspace(0, 2*np.pi, n_atoms, endpoint=False)
   atom_x = np.cos(angles)
   atom_y = np.sin(angles)
   atom_z = np.zeros(n_atoms)

   atom_x = np.append(atom_x, 0)
   atom_y = np.append(atom_y, 0)
   atom_z = np.append(atom_z, 0)

   gplt.figure("Molecular Structure", figsize=(9, 7))

   # Bonds (connections)
   for i in range(n_atoms):
       j = (i + 1) % n_atoms
       gplt.plot3d([atom_x[i], atom_x[j]],
                   [atom_y[i], atom_y[j]],
                   [atom_z[i], atom_z[j]],
                   'gray', linewidth=1.5, alpha=0.5)
       gplt.plot3d([0, atom_x[i]], [0, atom_y[i]], [0, atom_z[i]],
                   'gray', linewidth=1, alpha=0.3)

   # Atoms as spheres
   gplt.scatter3d(atom_x[:n_atoms], atom_y[:n_atoms], atom_z[:n_atoms],
                  color='red', s=20, label='Ring atoms')
   gplt.scatter3d([0], [0], [0], color='blue', s=25, label='Center')

   gplt.title('Molecular Ring Structure')
   gplt.legend()
   gplt.show()

Example 3: Seismic Wave Propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulate expanding spherical waves:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   rng = np.random.default_rng(42)

   # Wave propagation from epicenter
   n = 100000
   theta = rng.uniform(0, 2*np.pi, n)
   phi = rng.uniform(0, np.pi, n)

   gplt.figure("Seismic Waves", figsize=(10, 8))

   for r, time in [(2.0, '0s'), (2.3, '0.3s'), (2.6, '0.6s'), (3.0, '1.0s')]:
       x = r * np.sin(phi) * np.cos(theta)
       y = r * np.sin(phi) * np.sin(theta)
       z = r * np.cos(phi)

       # Add noise to wavefront
       z += 0.1 * np.sin(5*theta) * np.sin(5*phi)

       amplitude = np.exp(-(r - 2.0)**2 / 0.2)
       gplt.scatter3d(x, y, z, s=1.5, alpha=0.4*amplitude,
                      c=amplitude, cmap='hot',
                      elev=20, azim=-30, label=f'Wave @ {time}')

   gplt.scatter3d([0], [0], [0], color='black', s=30, label='Epicenter')
   gplt.title('3D Seismic Wave Propagation')
   gplt.legend()
   gplt.show()

Best Practices Summary
^^^^^^^^^^^^^^^^^^^^^

**For Optimal Performance:**

1. **Point Clouds**: Use alpha < 0.5, point size < 2, consider SSAO rendering
2. **Meshes**: Use stride parameters to reduce triangle count
3. **Vector Fields**: Subsample sparse grids (use 10-20% of points)
4. **Mixed 3D/2D**: Keep 2D overlays minimal
5. **Large Datasets**: Profile with your hardware, adjust quality settings

GPU Memory Considerations:

.. code-block:: python

   # Check if SSAO (Screen Space Ambient Occlusion) is available
   gplt.figure("SSAO Figure", figsize=(10, 7), ssao=True)
   # SSAO adds visual quality but increases GPU load

See Also
--------

- :doc:`basic-plotting` for 2D plotting fundamentals
- :doc:`advanced-features` for rendering options and effects
- API Reference: ``glplot.pyplot`` for detailed function signatures
