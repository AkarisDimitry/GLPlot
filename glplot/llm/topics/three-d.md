# 3-D plots

summary: 3-D scatter, surfaces, bars, volumes and vector fields; the camera; what differs from mplot3d.

Make a 3-D figure and draw into it:

```python
plt.figure("Cloud", figsize=(8, 6), projection="3d")     # or plt.scatter3d(...) on any figure
plt.scatter3d(x, y, z, c=z, cmap="plasma", s=3, elev=20, azim=-50)
plt.xlabel("x"); plt.ylabel("y"); plt.zlabel("z")
plt.savefig("cloud.png")
```

Grid cells can be 3-D too: `plt.subplots(1, 2, subplot_kw={"projection": "3d"})` or
`plt.subplot(projection="3d")`. `projection="polar"` is **not** supported (use `plt.polar(theta, r)`).

## The 3-D functions

`scatter3d`, `plot3d` (line), `plot_surface(X, Y, Z)`, `plot_wireframe`, `mesh3d`, `bar3d`, `quiver3d`,
`voxels`, `volume3d(x, y, z, values, threshold=...)`. In a 3-D figure `plt.plot(x, y, z)` and
`plt.scatter(x, y, z)` also draw in 3-D.

## Camera and axes

* `plt.view_init(elev=30, azim=-60)` -- angles in degrees (also `elev=`/`azim=` arguments on most 3-D calls).
* `plt.view_preset(...)` for standard views; `plt.auto_rotate(...)` for a turntable in the live window.
* `plt.zlabel`, `plt.zlim`, `plt.zscale`, `plt.zticks`, `plt.set_box_aspect`.
* In the window: drag to rotate, wheel to zoom.

## Good to know

* A live window handles millions of 3-D points; the headless export is matplotlib and is much slower
  for very large clouds -- subsample for exports, or look at it live.
* SSAO depth shading (`plt.figure(ssao=True)`) is a live-window effect.
* Surfaces want `X, Y, Z` as 2-D grids (use `np.meshgrid`).
