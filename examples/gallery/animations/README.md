# GLPlot Animated Examples

Sixteen animated examples, each built with `glplot.animation.FuncAnimation` and exported
to a GIF. Run every script and write GIFs into `examples/gallery/animations/results`:

```bash
python examples/gallery/animations/run_animations.py
```

The PES example can export through either rendering path:

```bash
# Same OpenGL geometry, depth testing, shaders, and marker sizing as the GUI.
python examples/gallery/animations/16_pes_particle_dynamics.py --renderer opengl

# Portable CPU fallback; works without an OpenGL display/context.
python examples/gallery/animations/16_pes_particle_dynamics.py --renderer matplotlib
```

Its default, `--renderer auto`, tries the invisible OpenGL context first and emits a warning
before falling back to the matplotlib renderer when OpenGL is unavailable.

An animated frame re-renders from scratch on every call, so these deliberately use much
smaller per-frame point counts than the static gallery's "millions of points" examples —
see each script's own comments for the reasoning. Grid/field-based animations (the
reaction-diffusion field, the fractal zoom, the expanding density cloud) get away with
representing far more data per frame than the literal-scatter ones, since their cost scales
with grid resolution rather than point count.

## Preview

Click any animation to open its script.

<table>
<tr>
<td align="center" width="25%"><a href="01_orbiting_star_cluster.py"><img src="results/01_orbiting_star_cluster.gif" alt="Orbiting star cluster" width="200"/></a><br><sub>01 · Orbiting star cluster</sub></td>
<td align="center" width="25%"><a href="02_traveling_wave_family.py"><img src="results/02_traveling_wave_family.gif" alt="Traveling wave family" width="200"/></a><br><sub>02 · Traveling wave family</sub></td>
<td align="center" width="25%"><a href="03_spectrum_analyzer_bars.py"><img src="results/03_spectrum_analyzer_bars.gif" alt="Spectrum analyzer bars" width="200"/></a><br><sub>03 · Spectrum analyzer bars</sub></td>
<td align="center" width="25%"><a href="04_reaction_diffusion_field.py"><img src="results/04_reaction_diffusion_field.gif" alt="Reaction-diffusion field" width="200"/></a><br><sub>04 · Reaction-diffusion field</sub></td>
</tr>
<tr>
<td align="center" width="25%"><a href="05_advected_flow_particles.py"><img src="results/05_advected_flow_particles.gif" alt="Advected flow particles" width="200"/></a><br><sub>05 · Advected flow particles</sub></td>
<td align="center" width="25%"><a href="06_expanding_density_cloud.py"><img src="results/06_expanding_density_cloud.gif" alt="Expanding density cloud" width="200"/></a><br><sub>06 · Expanding density cloud</sub></td>
<td align="center" width="25%"><a href="07_fractal_zoom.py"><img src="results/07_fractal_zoom.gif" alt="Fractal zoom" width="200"/></a><br><sub>07 · Fractal zoom</sub></td>
<td align="center" width="25%"><a href="08_rose_curve_trails.py"><img src="results/08_rose_curve_trails.gif" alt="Rose curve trails" width="200"/></a><br><sub>08 · Rose curve trails</sub></td>
</tr>
<tr>
<td align="center" width="25%"><a href="09_rotating_3d_starfield.py"><img src="results/09_rotating_3d_starfield.gif" alt="Rotating 3D starfield" width="200"/></a><br><sub>09 · Rotating 3D starfield</sub></td>
<td align="center" width="25%"><a href="10_rippling_3d_surface.py"><img src="results/10_rippling_3d_surface.gif" alt="Rippling 3D surface" width="200"/></a><br><sub>10 · Rippling 3D surface</sub></td>
<td align="center" width="25%"><a href="11_growing_3d_bar_city.py"><img src="results/11_growing_3d_bar_city.gif" alt="Growing 3D bar city" width="200"/></a><br><sub>11 · Growing 3D bar city</sub></td>
<td align="center" width="25%"><a href="12_force_directed_network.py"><img src="results/12_force_directed_network.gif" alt="Force-directed network" width="200"/></a><br><sub>12 · Force-directed network</sub></td>
</tr>
<tr>
<td align="center" width="25%"><a href="13_radar_sweep.py"><img src="results/13_radar_sweep.gif" alt="Radar sweep" width="200"/></a><br><sub>13 · Radar sweep</sub></td>
<td align="center" width="25%"><a href="14_streamgraph_flow.py"><img src="results/14_streamgraph_flow.gif" alt="Streamgraph flow" width="200"/></a><br><sub>14 · Streamgraph flow</sub></td>
<td align="center" width="25%"><a href="15_3d_fireworks.py"><img src="results/15_3d_fireworks.gif" alt="3D fireworks" width="200"/></a><br><sub>15 · 3D fireworks</sub></td>
<td align="center" width="25%"><a href="16_pes_particle_dynamics.py">16_pes_particle_dynamics.py</a><br><sub>16 · PES particle dynamics (run it to write the GIF)</sub></td>
</tr>
</table>

## Details

Ordered examples:

1. `01_orbiting_star_cluster.py` - a 20,000-star spiral galaxy winding its arms tighter
   under differential (Keplerian-like) rotation, colored by radius on a neon-plasma palette.
2. `02_traveling_wave_family.py` - a 24,000-line-per-frame `plot_lines(..., cmap=...)`
   density diagram of two interfering wave trains, breathing from a single bright band into
   a crossing "X" and back.
3. `03_spectrum_analyzer_bars.py` - a 64-band neon audio-spectrum-analyzer / equalizer, bars
   colored green-to-red by loudness with decaying peak-hold caps.
4. `04_reaction_diffusion_field.py` - a Gray-Scott reaction-diffusion simulation on a 170x170
   grid, five seed droplets growing into a coral-like Turing pattern.
5. `05_advected_flow_particles.py` - 4,500 tracer particles genuinely RK4-advected through a
   time-varying two-vortex-plus-drain velocity field, with a quiver overlay of the field.
6. `06_expanding_density_cloud.py` - a 300,000-fragment explosion debris cloud, binned live
   into a 160x160 density heatmap as it expands and blurs outward.
7. `07_fractal_zoom.py` - a Mandelbrot seahorse-valley zoom, escape-time grid recomputed and
   deepened every frame with a slowly hue-cycling turbo palette.
8. `08_rose_curve_trails.py` - a breathing rose (rhodonea) curve traced by a glowing comet
   head dragging a 1,200-point rainbow trail that fades in size, opacity, and hue with age.
9. `09_rotating_3d_starfield.py` - an orbiting camera circling a 6,500-star two-armed dwarf
   spiral galaxy (bulge + logarithmic-spiral disk), colored by galactocentric radius.
10. `10_rippling_3d_surface.py` - two water drops landing on a still pond, their
    dispersion-relation wave packets interfering as they cross, under a slowly orbiting camera.
11. `11_growing_3d_bar_city.py` - a 484-building 3D bar-city skyline whose rooftops pulse
    under two orbiting, ringing wave sources (like moving loudspeakers), height-colored on
    `inferno` with a slow camera orbit.
12. `12_force_directed_network.py` - a 160-node, 317-edge scale-free network laid out live by
    a real Fruchterman-Reingold force simulation, untangling from a tangled seed into a
    hub-and-spoke structure, nodes sized/colored by degree.
13. `13_radar_sweep.py` - a rotating PPI radar beam sweeping past 650 contacts on a manually
    computed polar display, with a glow wedge and per-contact afterglow that decays between
    passes.
14. `14_streamgraph_flow.py` - a 10-band streamgraph of a synthetic ocean-buoy swell
    spectrum, stacked with a centred "wiggle" baseline and a warm-to-cool hue sweep,
    scrolling through a slowly wandering storm event.
15. `15_3d_fireworks.py` - four staged 3D firework shells over a dark bay, each a
    physically-modeled particle burst that flashes white at ignition and fades from full hue
    to black as it burns out, under a slowly panning camera.
16. `16_pes_particle_dynamics.py` - 24 high-temperature Langevin particles crossing a
    double-well potential-energy surface, with short motion trails, thermal coordinate/energy
    traces, and a 3D camera that completes a full orbit around the PES. Data generation and
    plotting are separate functions so either stage can be reused independently.

See the [main gallery](../README.md) for GLPlot's static examples.
