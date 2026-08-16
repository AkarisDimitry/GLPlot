import numpy as np

import glplot.pyplot as plt

# Small-multiples dashboard: four independent 2D views of one synthetic
# regional weather-station network, each panel a single dominant visual
# motif (density field, scatter cloud, line family, bar chart) rather than
# the old SSAO on/off pair, whose two halves read as visually identical.
#
# GLPlot's multi-panel chrome (title/xlabel/ylabel/legend) is a single
# figure-wide value applied only to whichever panel is active when the file
# is saved -- there is no per-panel title/label storage yet. So the overall
# headline lives in a dedicated banner panel (a real set_title(), made active
# last) and each content panel gets its own in-plot caption (title + axis
# meaning + units) via a text() callout instead of set_title()/set_xlabel().
# A small invisible marker above each panel's real data reserves clean
# headroom for that caption without GLPlot's autoscale clipping it against
# the plotted data.
rng = np.random.default_rng(41)

n_stations = 80


def _lerp(t: float, c0, c1):
    return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(4))


def _caption(ax, x, y, text, fontsize=12):
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=(0.08, 0.08, 0.08, 1.0),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.78, edgecolor="none"),
    )


# ---- Panel A data: regional temperature field (density) ---------------------
grid_n = 200
gx = np.linspace(0.0, 120.0, grid_n)  # km, easting
gy = np.linspace(0.0, 90.0, grid_n)  # km, northing
X, Y = np.meshgrid(gx, gy)
temp_field = (
    14.0
    + 6.0 * np.exp(-(((X - 40.0) ** 2 + (Y - 30.0) ** 2) / 620.0))
    + 4.5 * np.exp(-(((X - 92.0) ** 2 + (Y - 62.0) ** 2) / 480.0))
    + 2.0 * np.sin(X / 14.0) * np.cos(Y / 18.0)
)

# ---- Panel B data: station humidity/pressure readings (scatter cloud) -------
# Doubled the reading count and widened the third-quantity spread (reading_temp)
# versus the original pass, specifically so the cloud reads as a deliberately
# dense, richly-colored mass rather than a thin scatter of dots -- "turbo"
# (a high-contrast rainbow-like colormap) then pushes that spread into visibly
# distinct hues from one end of the cloud to the other.
n_readings = 90_000
humidity = np.clip(rng.normal(60.0, 15.0, n_readings), 8.0, 97.0)  # %
pressure_anom = -0.4 * (humidity - 60.0) + rng.normal(0.0, 5.0, n_readings)  # hPa
reading_temp = 20.0 - 0.22 * (humidity - 60.0) + rng.normal(0.0, 4.5, n_readings)  # deg C

# ---- Banner data: the 80 stations' actual positions across the same domain --
# panel A's temperature field is drawn over -- so the banner becomes a real
# site map (colored by each station's own mean daily temperature) instead of
# a caption with nothing plotted on it.
station_x = rng.uniform(0.0, 120.0, n_stations)
station_y = rng.uniform(0.0, 90.0, n_stations)

# ---- Panel C data: 24 h temperature trace per station (line family) ---------
hours = np.linspace(0.0, 24.0, 120)
station_mean = rng.normal(15.0, 4.0, n_stations)
station_amp = np.abs(rng.normal(6.0, 2.0, n_stations)) + 1.5
station_phase = rng.uniform(0.0, 2.0, n_stations)
station_traces = station_mean[:, None] + station_amp[:, None] * np.sin(
    (hours[None, :] - 8.0 - station_phase[:, None]) / 24.0 * 2.0 * np.pi
)
station_traces = station_traces + rng.normal(0.0, 0.5, station_traces.shape)

# ---- Panel D data: precipitation across all 80 stations, half-hourly bins ---
# Finer time resolution (48 half-hour bins instead of 24 hourly ones) plus a
# second dimension folded in: precipitation is simulated *per station* first
# (a diurnal envelope, a per-station wet/dry bias, and per-reading noise), and
# only then averaged down to the bar heights -- so the yerr= error bars below
# are the real station-to-station spread at each bin, not a decorative flat
# value.
n_bins_d = 48
bin_width_d = 24.0 / n_bins_d  # 0.5 h
hour_bins = (np.arange(n_bins_d) + 0.5) * bin_width_d  # bin-center hours
diurnal_envelope = np.abs(1.0 + 2.4 * np.exp(-((hour_bins - 16.0) ** 2) / 20.0))  # mm
station_precip_bias = rng.normal(0.0, 0.35, n_stations)  # per-station wet/dry bias
station_precip = (
    diurnal_envelope[None, :]
    + station_precip_bias[:, None]
    + rng.normal(0.0, 0.6, (n_stations, n_bins_d))  # per-reading noise
)
station_precip = np.clip(station_precip, 0.0, None)
precip_mean = station_precip.mean(axis=0)  # bar heights
precip_std = station_precip.std(axis=0)  # yerr -- spread across the 80 stations

plt.figure("Gallery - Weather Network Dashboard", figsize=(13.5, 8.5))
shape = (5, 2)
banner = plt.subplot2grid(shape, (0, 0), rowspan=1, colspan=2)
panel_a = plt.subplot2grid(shape, (1, 0), rowspan=2, colspan=1)
panel_b = plt.subplot2grid(shape, (1, 1), rowspan=2, colspan=1)
panel_c = plt.subplot2grid(shape, (3, 0), rowspan=2, colspan=1)
panel_d = plt.subplot2grid(shape, (3, 1), rowspan=2, colspan=1)

# Panel A -- density field
panel_a.contourf(X, Y, temp_field, levels=28, cmap="inferno")
panel_a.scatter([4.0], [90.0 * 1.14], s=0.01, alpha=0.0)  # reserve caption headroom
_caption(panel_a, 4.0, 90.0 * 1.04, "Density -- regional temperature field (deg C)")

# Panel B -- dense scatter cloud. Smaller, lower-alpha points than the
# original pass, but nearly 4x as many raw readings underneath them: the
# per-point restraint is what lets the extra density read as a soft, glowing
# mass instead of a muddy blob, while "turbo" keeps the humidity/temperature
# gradient across it high-contrast end to end.
panel_b.scatter(humidity, pressure_anom, c=reading_temp, cmap="turbo", s=1.6, alpha=0.35)
xb_min, xb_max = float(humidity.min()), float(humidity.max())
yb_min, yb_max = float(pressure_anom.min()), float(pressure_anom.max())
panel_b.scatter([xb_min], [yb_max + 0.18 * (yb_max - yb_min)], s=0.01, alpha=0.0)
_caption(
    panel_b,
    xb_min + 0.02 * (xb_max - xb_min),
    yb_max + 0.05 * (yb_max - yb_min),
    f"Scatter -- {n_readings:,} readings: humidity (%) vs pressure anomaly (hPa), "
    "colored by reading temperature",
)

# Panel C -- line family
c_cool = (0.16, 0.38, 0.85, 0.4)
c_warm = (0.95, 0.55, 0.12, 0.4)
station_range = station_mean.max() - station_mean.min()
t_norm = (station_mean - station_mean.min()) / (station_range if station_range > 0 else 1.0)
for i in range(n_stations):
    panel_c.plot(
        hours, station_traces[i], color=_lerp(float(t_norm[i]), c_cool, c_warm), linewidth=0.9
    )
yc_min, yc_max = float(station_traces.min()), float(station_traces.max())
panel_c.scatter([0.5], [yc_max + 0.2 * (yc_max - yc_min)], s=0.01, alpha=0.0)
_caption(
    panel_c,
    0.5,
    yc_max + 0.06 * (yc_max - yc_min),
    f"Lines -- {n_stations} station traces: temperature (deg C) vs hour (h)",
)

# Panel D -- bar chart, 48 half-hourly bins with yerr= station-spread error bars
panel_d.bar(
    hour_bins,
    precip_mean,
    width=bin_width_d * 0.85,
    color=(0.93, 0.55, 0.16, 0.88),
    yerr=precip_std,
    ecolor=(0.32, 0.16, 0.03, 0.85),
    capsize=bin_width_d * 0.3,
)
yd_max = float((precip_mean + precip_std).max())
panel_d.scatter([-0.4], [yd_max * 1.32], s=0.01, alpha=0.0)
_caption(
    panel_d,
    -0.4,
    yd_max * 1.1,
    f"Bars -- mean precip per {bin_width_d:g} h bin (mm), "
    f"error bars = spread across {n_stations} stations",
)

# Banner -- was a caption strip with nothing plotted on it (just a fill and a
# line of text); now a real fifth panel, a site map of the 80 stations, colored
# by each one's own mean daily temperature (the same quantity panel C's line
# family plots the full 24h trace of) so it visibly connects to the rest of the
# dashboard instead of only summarizing it in prose. GLPlot's title/xlabel/
# ylabel chrome is a single figure-wide value drawn only on whichever panel is
# active at export time, so this panel (made active last) still carries the
# overall title via a real, properly centred set_title().
banner.scatter(
    station_x,
    station_y,
    c=station_mean,
    cmap="inferno",
    s=110,
    edgecolors=(1.0, 1.0, 1.0, 0.9),
    linewidths=1.1,
)
banner.set_xlabel("Easting (km)")
banner.set_ylabel("Northing (km)")
# A corner caption (same _caption() helper the other four panels use) rather
# than text floating above the axes: that collided with the real set_title()
# below, which matplotlib also draws just above the axes box.
_caption(
    banner,
    2.0,
    90.0 * 0.90,
    f"{n_stations} station map, colored by mean daily temperature (deg C)",
    fontsize=14,
)
banner.set_title("Regional Weather-Station Network -- One Synthetic Day")

plt.savefig("examples/gallery/results/16_ssao_comparison.png")
