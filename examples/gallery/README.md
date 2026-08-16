# GLPlot Example Gallery

Run every gallery script and write PNG previews into `examples/gallery/results`:

```bash
python examples/gallery/run_gallery.py
```

Ordered examples:

1. `01_line_plot.py` - 10,072-line oscilloscope ensemble styled with `plot_style("chalk")`, each trace a distinct pastel hue on a chalkboard background.
2. `02_scatter_fill.py` - 10-million-point spiral scatter (one vectorized `ax.scatter()` call) with turbo color banding from center to rim.
3. `03_bar_hist.py` - million-sample histogram with a bar overlay.
4. `04_line_family_density.py` - one 20-million-line family, rendered four times in four unrecognizably different styles (whiteboard, chalkboard, hand-drawn notebook, neon) via `plot_lines(..., cmap=...)` and composited into one figure.
5. `05_guides_and_colormap.py` - CPU-baked accumulate-then-resolve density field across 16 overlapping population blobs, with axis guides through the origin.
6. `06_signal_tools.py` - zoomed-in 2 s window of a voltage trace with `step`, `errorbar`, and event `stem`.
7. `07_projected_3d_cloud.py` - simulated circumstellar dust halo, 1,000,000 depth-shaded grains filling a tight spherical frame.
8. `08_vector_field_quiver.py` - arrows, RK4-advected tracer-particle streaks, and a vector field over a matrix.
9. `09_large_matrix_heatmap.py` - large procedural matrix heatmap.
10. `10_massive_hist2d_density.py` - 10-million-hit 2D density histogram, zoomed to the populated core.
11. `11_contour_pcolormesh_field.py` - contour and contourf on a 520 x 520 field (`pcolormesh()` avoided here: its headless preview reconstruction is a known bug).
12. `12_surface_wireframe_bar3d.py` - interference surface (`plot_surface` + `plot_wireframe`), its amplitude flattened onto the floor as a dense point-cloud projection.
13. `13_volumetric_nebula.py` - volumetric emission nebula, 1,750,000 points shaded by local emission intensity in a spiral density field.
14. `14_bar3d_hex_box_city.py` - mixed square and hexagonal 3D bars forming a downtown skyline, height-colored gold-to-red (N = 361 towers).
15. `15_vector_field_3d.py` - swirling 3D jet velocity field, ~5,300 vectors traced through a 420k-sample aerosol cloud.
16. `16_ssao_comparison.py` - small-multiples 2D dashboard: an 80-station site-map banner, a density field, a 90k-point scatter cloud, an 80-station line family, and a 48-bin half-hourly bar chart with `yerr=` error bars for the spread across stations -- five views of one synthetic weather network.
17. `17_square_bars3d.py` - square 3D bars with edges and SSAO, a downtown skyline mound on a fire-red height scale (N = 1,369 bars).
19. `19_turbulent_vector_field_3d.py` - massive 3D turbulent flow: a 950k-point volumetric core, an adaptive vector lattice, and 16 braided stream traces.
20. `20_categorical_bar_and_spines.py` - `bar()` with string categories, `yerr=`, and `ax.spines[...].set_visible(False)`; each bar is built from a 650k-point jittered scatter column (2.6M points total), so the silhouette is visibly made of raw samples.
21. `21_date_axis_and_text_bbox.py` - `datetime.date` values plotted directly, manual date-tick labels, and `text(bbox=...)`, on a 16-ridge rainbow joyplot of a simulated 3-million-reading sensor stream, every ridge fully neon-glowing over the faint raw-reading scatter it was averaged from.
22. `22_log_scale_plots.py` - `semilogy()` under ~3 million individually simulated Monte-Carlo decay events (non-homogeneous Poisson process), plotted as a dense point cloud around the theoretical decay curve.
23. `23_symlog_scale.py` - `yscale('symlog', linthresh=...)`: a 2-million-sample damped LC ringdown, glowing green core traced by its magenta decay envelope, with raw noisy measurements as a magenta grain texture that thickens as the signal decays into the noise floor.
24. `24_log_scale_bar_chart.py` - `bar()` with a real log y-axis across 1,500 Zipf-law-ranked bars (a spectrum-analyzer skyline), each topped with a jittered swarm of repeated-measurement samples.
25. `25_log_scale_contourf.py` - `contourf()` on a real log x-axis, with 1.5 million simulated detector hits sampled from the field's own distribution.
26. `26_logit_scale.py` - `yscale('logit')`: a pastel chalk-gold dose-response fit glowing through a soft mint/coral cloud of 2.4 million simulated Bernoulli trials, on a chalkboard stage (`plot_style("chalk")`).
27. `27_inset_axes_image.py` - `ax.inset_axes(...)` with true-color `imshow()`: a 2.32-million-point wide-field scatter with three inset panels rasterizing full-detail crops of its own clusters.
28. `28_chladni_wave_animation.py` - animated standing-wave interference pattern (`glplot.animation.FuncAnimation`) sweeping through vibrational mode numbers with a live camera zoom in/out, exported to a GIF via `ani.save(...)`; styled with `plot_style("blueprint")`.
