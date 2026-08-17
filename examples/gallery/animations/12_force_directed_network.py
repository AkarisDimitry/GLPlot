"""Force-directed network layout -- a scale-free graph untangling into structure.

Two classic pieces of network science, run back to back. First, a graph is grown by
*preferential attachment* (the Barabasi-Albert model): starting from a small seed clique,
each new node arrives with a couple of edges and picks which existing nodes to attach to
with probability proportional to how many edges that node already has -- "the rich get
richer." Run that for a few hundred nodes and the result is a *scale-free* network: most
nodes end up with only a handful of edges, while a small number of early, lucky nodes
become high-degree hubs -- the same heavy-tailed shape seen in citation networks, the web's
hyperlink graph, and airline route maps.

Second, that graph -- which has no geometry of its own, only *who connects to whom* -- is
laid out in the plane by the Fruchterman-Reingold force-directed algorithm: every pair of
nodes repels like charged particles (``F ~ k^2 / d``, so crowding is expensive), while every
*edge* pulls its two endpoints together like a spring (``F ~ d^2 / k``, so long edges are
expensive too). Iterating that push-pull under a shrinking "temperature" cap on how far any
node may move per step is a physical annealing process: it starts as a tight, tangled
knot -- exactly what a graph looks like with no layout information at all, here seeded as a
small random cluster -- and, step by step, relaxes into the hub-and-spoke structure that was
latent in the connectivity the whole time. The hubs the attachment process created are
*rediscovered* by the physics: nothing tells the layout which nodes are hubs, but the ones
with the most springs pulling on them are exactly the ones the repulsion has to fight
hardest to keep separated, so they visibly end up as the well-spaced anchors other nodes
orbit.

Every node's position is genuinely re-integrated each frame (a vectorised numpy pairwise
repulsion plus a per-edge spring attraction, several sub-steps per rendered frame) -- this is
the actual physical simulation settling on screen, not a pre-baked interpolation between a
random start and a fixed end layout. Edges (a few hundred, modest enough for literal
segments) are drawn as ONE ``plot()`` call whose x/y arrays are NaN-separated per edge --
cheap, and it's exactly the pattern this gallery's own ``quiver()`` shaft-batching already
relies on. Nodes are drawn as several ``scatter()`` calls, one per degree band, so marker
*size* -- which the headless GIF export only honours as a single scalar per call, not a
per-point array -- still reads as "hub nodes are bigger," while colour stays a fully
continuous read of degree via a shared colormap normalisation across every band.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps

import glplot.animation as animation
import glplot.pyplot as plt

FRAMES = 84
SUBSTEPS = 4  # Fruchterman-Reingold relaxation iterations folded into every rendered frame
N_NODES = 160
M_EDGES = 2  # new-node attachment count in the Barabasi-Albert construction
SEED = 11

AREA_SIDE = 9.0  # characteristic layout box side; sets the FR spring constant k
LIM = 7.0  # fixed view half-width/height (measured: the run never exceeds ~6.1)


def build_scale_free_graph(n: int, m: int, rng: np.random.Generator):
    """Barabasi-Albert preferential attachment: ``edges (E, 2)`` and per-node ``degree``.

    ``stubs`` holds one entry per existing edge-endpoint, so sampling uniformly from it is
    exactly sampling a node with probability proportional to its current degree -- the
    "rich get richer" rule, without needing to track or renormalise probabilities by hand.
    """
    m0 = m + 1  # seed clique size
    edges = []
    degree = np.zeros(n, dtype=np.int64)
    for i in range(m0):
        for j in range(i + 1, m0):
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1
    stubs = []
    for node in range(m0):
        stubs.extend([node] * degree[node])
    for new in range(m0, n):
        targets: set[int] = set()
        while len(targets) < m:
            targets.add(stubs[rng.integers(len(stubs))])
        for t in targets:
            edges.append((new, t))
            degree[new] += 1
            degree[t] += 1
            stubs.append(new)
            stubs.append(t)
    return np.asarray(edges, dtype=np.int64), degree


def relax(
    pos: np.ndarray, src: np.ndarray, dst: np.ndarray, k: float, temperature: float
) -> np.ndarray:
    """One Fruchterman-Reingold step: all-pairs repulsion plus per-edge spring attraction.

    Every node's net displacement is capped at ``temperature`` -- the "annealing" that
    keeps the simulation stable even when two nodes start on top of each other (repulsion
    would otherwise blow up as ``1/d`` at ``d -> 0``) and, cooled across the run, is what
    turns an initially chaotic tangle into a settled layout instead of a permanent jitter.
    """
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 2): every pairwise offset at once
    dist = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(dist, np.inf)  # a node does not repel itself
    dist = np.maximum(dist, 1e-3)
    repulsion = (diff / dist[..., None]) * (k * k / dist)[..., None]
    disp = repulsion.sum(axis=1)

    d = pos[src] - pos[dst]
    dist_e = np.maximum(np.sqrt((d * d).sum(-1)), 1e-3)
    attraction = (d / dist_e[:, None]) * (dist_e * dist_e / k)[:, None]
    np.subtract.at(disp, src, attraction)  # pulls src toward dst
    np.add.at(disp, dst, attraction)  # and dst toward src

    length = np.maximum(np.sqrt((disp * disp).sum(-1)), 1e-9)
    step = np.minimum(length, temperature)
    return pos + disp / length[:, None] * step[:, None]


rng = np.random.default_rng(SEED)
edges, degree = build_scale_free_graph(N_NODES, M_EDGES, rng)
src, dst = edges[:, 0], edges[:, 1]
N_EDGES = len(edges)
DEG_MIN, DEG_MAX = int(degree.min()), int(degree.max())

K = AREA_SIDE / np.sqrt(N_NODES)  # FR's characteristic spring/repulsion length scale

# Seeded as a tight random cluster -- deliberately "no layout information yet" -- so the
# very first frames genuinely read as a tangled knot rather than an already-loose scatter.
pos = rng.normal(scale=0.55, size=(N_NODES, 2))

TOTAL_STEPS = FRAMES * SUBSTEPS
T0, T_MIN = K * 0.10, K * 0.006  # measured: this schedule spreads ~80% of the unfolding
DECAY = (T_MIN / T0) ** (1.0 / TOTAL_STEPS)  # over the first half of the run, then settles

# NaN-separated scratch buffers for drawing every edge as one polyline: [a0, b0, NaN, a1,
# b1, NaN, ...]. Positions get overwritten every frame; the NaN gaps never move.
edge_x = np.full(3 * N_EDGES - 1, np.nan, dtype=np.float64)
edge_y = np.full(3 * N_EDGES - 1, np.nan, dtype=np.float64)

# Degree -> marker-size band, geometrically spaced so a handful of high-degree hubs (a
# scale-free tail) each land in their own band instead of being swamped by the bulk of
# degree-2/3 nodes. See the module docstring for why size has to be banded rather than
# continuous under headless export.
N_BANDS = 6
band_edges = np.geomspace(DEG_MIN, DEG_MAX + 1, N_BANDS + 1)
band_sizes = 9.0 + 55.0 * (np.arange(N_BANDS) / max(N_BANDS - 1, 1)) ** 1.4

HUB_COUNT = 4
hub_idx = np.argsort(degree)[-HUB_COUNT:]
HUB_RGB = np.asarray(colormaps["plasma"](1.0))[:3]

fig = plt.figure("Gallery - Force-Directed Network", figsize=(8.0, 7.4))
plt.plot_style("blueprint")


def update(frame: int):
    global pos
    step0 = frame * SUBSTEPS
    for s in range(SUBSTEPS):
        temperature = T0 * (DECAY ** (step0 + s))
        pos = relax(pos, src, dst, K, temperature)

    xs, ys = pos[:, 0], pos[:, 1]
    edge_x[0::3] = xs[src]
    edge_x[1::3] = xs[dst]
    edge_y[0::3] = ys[src]
    edge_y[1::3] = ys[dst]

    plt.cla()
    plt.plot(edge_x, edge_y, color=(0.68, 0.86, 1.0, 0.30), linewidth=0.8)

    for b in range(N_BANDS):
        lo, hi = band_edges[b], band_edges[b + 1]
        mask = (degree >= lo) & (degree < hi if b < N_BANDS - 1 else degree <= hi)
        if not np.any(mask):
            continue
        plt.scatter(
            xs[mask],
            ys[mask],
            c=degree[mask],
            cmap="plasma",
            vmin=DEG_MIN,
            vmax=DEG_MAX,
            s=band_sizes[b],
            alpha=0.92,
            edgecolors=(1.0, 1.0, 1.0, 0.35),
            linewidths=0.5,
        )

    # A soft glow on the top hubs, fading in as the layout settles -- once the physics has
    # done its own job of spacing the hubs out, draw the eye to exactly the nodes the
    # attachment process (and now the layout) singled out as structurally important.
    settle = min(1.0, frame / (FRAMES * 0.55))
    if settle > 0.05:
        glow_rows = [(*HUB_RGB, 0.05 * settle)] * HUB_COUNT
        for i in range(1, 9):
            plt.scatter(xs[hub_idx], ys[hub_idx], c=glow_rows, s=14.0 * (1.22**i))

    # `xlim()`/`ylim()` alone are inert against the headless export's own autoscale-to-data
    # pass, which would otherwise reframe every single frame to the shrinking/growing
    # cluster's own extent -- exactly the camera jitter a fixed layout box should not have.
    # A pair of fully transparent corner points still count toward the data's bounding box
    # (autoscale reads coordinates, not alpha), which pins the view without being visible.
    plt.scatter([-LIM, LIM], [-LIM, LIM], c=[(0.0, 0.0, 0.0, 0.0)] * 2, s=0.001)
    plt.set_aspect("equal")
    plt.xlim(-LIM, LIM)
    plt.ylim(-LIM, LIM)
    plt.title(
        f"Scale-free network layout settling -- N={N_NODES}, E={N_EDGES}, T={temperature:.3f}"
    )
    plt.xlabel("Layout x (arb. units)")
    plt.ylabel("Layout y (arb. units)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
ani.save("examples/gallery/animations/results/12_force_directed_network.gif", fps=20)
# plt.show()
