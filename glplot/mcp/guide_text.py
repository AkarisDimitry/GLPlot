# flake8: noqa: E501
# ruff: noqa: E501
"""The prose of the animation and interactive guide topics (see :mod:`glplot.mcp.guide`).

Kept apart from the code that assembles the guide because it is long and is edited as writing.
``ANIMATION_BODY`` is a ``str.format`` template: literal braces are doubled, and the placeholders
``{vocabulary}``, ``{formats}``, ``{max_frames}``, ``{max_stages}`` and ``{max_seconds}`` are
filled from the code that enforces them, so the guide cannot promise what the server refuses.
"""

ANIMATION_INTRO = """`animate(figure_id, ...)` makes properties of a figure follow **time**, in **stages**, and
writes the result as a GIF, MP4, WebM or numbered PNGs. You cannot watch an animation, so you are given
what you *can* check: a contact sheet (frames side by side, captioned with frame, time and stage), the
value of every variable per stage, and `issues` measured on the frames (nothing moves, the axes rescale,
the loop does not close, an abrupt jump).

## The loop to follow

1. Build the figure **as a still**, with everything in its final position: `create_figure`, data
   (`create_dataset`, `simulate`, `load_dataset`), `add_plot` / `add_function` / `add_field`.
2. **Pin the axes**: `set_axes(xlim=[...], ylim=[...])` (and `zlim`). Otherwise the view re-fits every
   frame and the picture zooms and jitters (`animate` warns: `xlim_moves`).
3. Say what varies: `variables` (+ `stages`) define the numbers over time; `bindings` say which
   property follows which variable.
4. **Preview** first: `animate(...)` *without* `output` returns the contact sheet. Look at it. Read `issues`.
5. Fix what is wrong, then repeat with `output="name.gif"` (or `.mp4`).
6. Give the user the file, or `export_script(figure_id, kind='animation')` for the Python behind it.
"""

ANIMATION_BODY = """## How it works

The figure is **rebuilt for every frame** from the calls that made it, with each bound property
replaced by its value at that frame. So any artist -- line, scatter, bar, field, 3-D surface, a
right-hand axis -- animates the same way, through the same checks as a still. Your figure is never
changed by `animate`. A binding that cannot be applied fails on frame 0, naming the artist and the reason.

### Time and variables

* Always available in expressions: `t` (0..1 over the animation), `time` (seconds), `frame`
  (0, 1, 2...), `n` (number of frames), `s` (0..1 *within the current stage*), `stage` (its number).
* `variables` are your own names. A **number** is a *parameter*: it holds that value until a stage
  moves it. An **expression string** is *derived* each frame from `t`, `time`, `s`, `stage` and the
  parameters: `{{"amp": 1, "phase": "2*pi*time", "w": "2*pi*f"}}`.
* `loop=true` makes `t` stop one step short of 1, so the last frame leads into the first. Give every
  moving quantity a whole number of periods (`2*pi*t`, `sin(2*pi*t)`) and the loop closes.
* Controls (`add_control`) are parameters too: a stage can sweep a slider.

### Stages (the multi-step part)

`stages=[{{"name", "duration" (s) | "frames", "ease", "animate": {{...}}}}, ...]` run one after another.
In `animate`, each variable can be:

| form | meaning |
|---|---|
| `"amp": [1, 3]` | from 1 to 3 over the stage |
| `"amp": 3` | to 3, from wherever it is now |
| `"amp": {{"from": 1, "to": 3, "ease": "in_out"}}` | with its own easing |
| `"k": "2 + 3*s"` | any expression of `s` (0..1 across the stage) |

A stage starts where the previous one left off and **ends exactly on its target**; afterwards the
value holds. A stage with no `animate` is a pause (derived variables keep running). Stages without a
length share what is left of the animation's `duration`.

{vocabulary}

### Bindings: what the variables drive

`bindings=[{{"target", "property", "value"}}]` (or `bind_property`, which makes one permanent).

| target | property is... |
|---|---|
| an artist id (`"a1"`) | **any argument of the call that drew it** (see `describe_figure` -> `recipe`): `color`, `linewidth`, `alpha`, `size`, `marker`, `x` / `y` / `z` (as a list), `reveal`, `trail`, `head`, `bins`, `cmap`, `vmin`, `vmax`, `x_min`, `x_max`; `parameters.NAME` reaches a function's or field's parameter (`parameters.phase`) |
| `"axes"` / `"axes:2"` | `title`, `xlabel`, `ylabel`, `xlim`, `ylim`, `zlim`, `elev`, `azim` (the 3-D camera), `legend`, `grid`, `y2lim`... |
| `"data:NAME"` | a **column** of that dataset, recomputed each frame from its other columns: `{{"target": "data:pts", "property": "x", "value": "lerp(cx, lx, m)"}}` |

A **value** is a number, an expression string, or one of:

* a list, one expression per element: `"xlim": ["0", "10*t"]`; `"x": ["5 + 4*sin(2*pi*t)"]`;
* `{{"colormap": "viridis" | ["#4c78a8", "#f58518"], "at": "t"}}` -- a colour moving along a ramp;
* `{{"choose": ["red", "green", "blue"], "index": "floor(3*t)"}}` -- one item of a list;
* `{{"format": "amp = {{amp:.2f}} at {{time:.1f}} s"}}` -- text with variables filled in (titles,
  labels). Only `{{name}}` and `{{name:.2f}}` are allowed; write a brace twice to get a literal one.

For a **text** property (`color`, `cmap`, `label`, `title`, `marker`, `linestyle`, `hatch`, `axis`) a plain
string is the text itself: `"color": "red"`, `"title": "Stage 2"`. Use `colormap`, `choose` or `format` to
make text change over time. Everything else takes numbers, so a plain string there is an expression.
A property the artist's call does not have is refused with a suggestion (`colour` -> `color`).

A column named like a variable (`t`!) wins inside `data:` expressions; use `time` or `frame` for the
clock there (`animate` says when this happens).

### Moving things, drawing things in

* `add_plot(..., reveal=0.3)` shows only the first 30% of the points; bind `reveal` to a variable and the
  curve **draws itself**. `trail=N` keeps only the last N points (a moving tail); `head=true` puts a dot on
  the newest point. They cut every series together (colours, sizes) and work in 3-D, where a line always
  shows at least its first two points (one point cannot make a line), so `reveal` may start at 0.
* A moving dot: bind `x` / `y` (`["10*t"]`), or reveal a trajectory with `head=true`.
* Growing bars, morphing points, changing data: a `data:` binding on the dataset the artist reads.
* A 3-D camera orbit: bind `azim` (and `elev`) on `"axes"`. Pin `xlim`, `ylim`, `zlim` first.
* A parameter sweep: `add_function(parameters={{"a": 1}})`, then bind `parameters.a`; or sweep a control.
* Real dynamics (pendulum, orbit, predator-prey, Lorenz): `simulate` solves the equations into a dataset;
  plot it with `reveal` + `head`.

## Step by step, like a manim explainer: stage `effects`

An explainer is a sequence of plays: draw the curve, write the formula, bring a dot in, point at the
term that matters, clear the stage. A stage's `effects` does that, all of its effects together:

`{{"name": "tangent", "duration": 1, "effects": {{"create": "a4", "write": ["a5", "a6"]}}}}`

| effect | what it does |
|---|---|
| `create` / `uncreate` | draws a curve in from its start / takes it back (line, scatter, step, area, `add_function`); an arrow grows; text is written. A kind that cannot be drawn in part (a field, bars) fades instead |
| `write` / `unwrite` | text letter by letter (a `$...$` formula appears whole) |
| `grow` | an arrow from its tail |
| `fade_in` / `fade_out` | opacity |
| `indicate` | turns yellow and grows, then comes back (manim's `Indicate`) |

* An artist brought in by an effect is **hidden until its stage**; one taken out stays hidden.
  Artists no effect names are on screen from the first frame (the axes, say).
* Effects ease with manim's `smooth` (here `smoother`) unless the stage gives `ease`. A stage can also
  `animate` variables, and other stages can move things while nothing appears (a point sliding
  along a curve, with `"text": {{"format": "slope = {{m:.2f}}"}}` updating beside it).
* Effects drive `alpha`, `reveal`, `color` and the size of the artists they name: do not bind those
  yourself as well (the call is refused). The reply's `effects` lists what each stage did.
* Pacing like the videos: 1-2 s per play, a short stage with nothing in it after each idea, and
  `hold_end=1` so the last picture stays.

**The look**: `create_figure(style="manim")` (or `set_style`) is manim's: a black page, BLUE_C for the
thing being explained and YELLOW_C to point at things, white serif text, `$...$` maths in Computer
Modern, axes that cross at the origin with arrow tips. `set_axes(grid=true)` gives a NumberPlane grid,
`legend=false` drops the legend (an explainer labels things in place).

**The pieces**: a title or caption that stays put, `add_annotation(kind="text", coords="axes", x=0.5,
y=1.05, align="center", fontsize=34)`; a formula, `text="$f'(x) = 2x$"`; a number that changes, bind
`text` to a `format`; a vector, `add_annotation(kind="arrow", x=0, y=0, dx=2, dy=1)` and bind `dx`,
`dy`; a dot, `add_plot(kind="scatter", x=[1], y=[1], size=16)` and bind `x`, `y`. The recipe
`explainer_manim` below is a complete one.

## Reading the reply

`frames`, `fps`, `duration_s`, `size_px`, `stages` (name, start, end), `variables` (first, last, min, max
-- check that each ends where you meant), `file` (path, bytes, codec), `sheet` (the frames shown), and
`issues`, each `{{severity, code, message, fix, scope}}`:

| code | meaning and fix |
|---|---|
| `static` (warning) | no frame differs: the bindings do nothing (wrong target id, a variable that never moves) |
| `xlim_moves` / `ylim_moves` (warning) | the view rescales between frames: `set_axes(xlim=..., ylim=...)` first |
| `loop_seam` (warning) | the last frame does not lead into the first: whole periods, and `loop=true` |
| `abrupt_change` (info) | one step is far bigger than its neighbours: ease it, or start the next stage where the last ended |
| `mostly_still` (info) | most steps show no change: shorten the stage that does nothing |
| `large_file` (suggestion) | a GIF over 25 MB: lower `scale`/`fps`, or write an .mp4 |
| `scope: figure` | what the still review found (missing labels...), as in `render_figure` |

`animate(frame=N)` renders one frame at full size (nothing is written): use it to inspect a detail.

## Output

* `output`: {formats}, or `"frames/f_%04d.png"` for one image per frame. Omit it to preview. Files go
  inside the server's `--root` and are never overwritten unless `overwrite=true`.
* GIF: 256 colours per frame, timing in hundredths of a second (30 fps plays as 33.3). MP4/WebM are
  far smaller and exact; they need `ffmpeg` on the PATH. `hold_start` / `hold_end` add seconds on the
  first/last frame; `repeat` is extra GIF plays (0 = forever); `scale` multiplies the resolution
  (default: about 960 px on the long side); `quality` is `draft`/`good`/`high` for video.
* Every frame has the same size: text is measured once over sample frames and one crop is used.

## Limits and cost

At most {max_frames} frames, {max_stages} stages and {max_seconds} s of rendering. A frame costs about
0.04-0.1 s in 2-D and 0.2-0.5 s in 3-D; the first frames are timed and an animation that would overrun
is refused up front with advice. Lower `fps` or `duration`, or preview a short version first.
"""

INTERACTIVE_BODY = """## Change what is drawn: `update_artist`

`update_artist(figure_id, artist_id, changes)` changes an existing artist without removing it. `changes`
holds arguments of the call that drew it (`describe_figure` shows each artist's `recipe`):

* `{"color": "tab:red", "linewidth": 3}` -- restyle;
* `{"y": "v_fit", "label": "fit"}` -- plot another column;
* `{"parameters.a": 4}` -- a dotted name reaches inside an object argument (a function's parameters);
* `{"color": null}` -- back to the default;
* `{}` -- redraw from the datasets as they are now (after you replaced one).

The artist keeps its id and its place in the stacking order. A change that cannot be drawn is refused
and the figure is left exactly as it was.

## Controls and bindings: a figure that depends on parameters

* `add_control(figure_id, name, kind, ...)`: `slider` (`min`, `max`, optional `step`, `integer`, `log`),
  `toggle` (true/false; 1/0 in expressions), `choice` (`options`; in expressions its value is the
  option's index) or `button` (`sets={"amp": 1}` applies values; without `sets` it resets every control).
* `bind_property(figure_id, target, property, value)`: make a property follow an expression of the
  controls, e.g. `bind_property("fig1", "a1", "parameters.amp", "amp")`. Targets and value forms are the
  ones `animate` uses (`get_guide('mcp-animation')`). `value=null` removes a binding.
* `set_control(figure_id, name, value)` moves a control and redraws the figure at it (a button is
  pressed without a value). `render_figure(figure_id, params={"amp": 2})` and `save_figure(params=...)`
  try values **without** changing the figure.
* `animate` can sweep a control in a stage (`"animate": {"amp": [0.5, 3]}`).
* `describe_figure` lists the controls and bindings.

Headless, a control is a named value you can set and scrub. For a real window with sliders, hand the
user `export_script(figure_id, kind="interactive")`.

## A second y axis, and hatching

`add_plot(..., axis="right")` draws against a second y axis on the right (line, scatter, step, bar, area,
errorbar). Name and range it with `set_axes(y2label="rate (1/s)", y2lim=[0, 50], y2scale="log")`. The left
axis fits its own data only. `hatch="//"` (also `"x"`, `"+"`, `"."`, `"o"`) patterns bar, barh, area and hist.

## `export_script`: the Python behind the figure

`export_script(figure_id, kind)` returns a script (as text; nothing is written to disk) built from the
calls that made the figure:

* `figure` (default): the figure as it is now, saved to a PNG with `plt.savefig`;
* `interactive`: the controls become `plt.slider`, `plt.checkbox`, `plt.dropdown`, `plt.button`, and a
  `build()` redraws the figure when one moves; `plt.show()` opens the window;
* `animation`: the last `animate` call as a `FuncAnimation`: the timeline as numpy arrays, an `update(i)`
  that redraws frame i, and `ani.save("fig1.gif", fps=FPS)`.

Datasets are rebuilt in the script from how they were made (expressions become numpy, `simulate` becomes
`solve_ivp`, a loaded file becomes `np.genfromtxt`); results of `analyze_dataset` are embedded as numbers
when small, and `notes` says when they are too big. The script needs only `glplot` (and `scipy` for
`simulate`). Give it to the user to keep and edit; unattended runs end with `savefig`, never `show`.
"""

VIDEO_BODY = """A whole explainer -- a title card, a few scenes that build an idea step by step, a recap -- is
**one figure per scene**, each animated and checked on its own, then **cut together** with
`compose_video`. This is how to make a complete explainer video in the manim style.

## 1. Plan

Write the script first: 3-6 scenes of 5-20 s, **one idea each**, in the order a viewer needs them.
For each scene list its plays (what appears, what moves, what is pointed at). Typical shape:

1. **Title card**: the question, written (`write`), maybe one formula fading in.
2. **Scenes**: build the picture (`create` the curve, `write` the formula, `fade_in` the dot), make
   it move (a stage that `animate`s a variable), point at the key term (`indicate`), turn one
   equation into the next (`transform`), clear what is done (`fade_out`).
3. **Recap**: the result on its own, held for a moment.

## 2. Build every scene the same way

* `create_figure(width=1280, height=720, style="manim")` -- the **same size for every scene**, so
  nothing is letterboxed. A text-only card: `set_axes(hide_axes=true, xlim=[0, 1], ylim=[0, 1])`
  and place text with `coords="axes"`.
* Pin the axes (`xlim`, `ylim`), `legend=false`, and label things in place with text.
* `animate(figure_id, stages=[...effects...])` **without output**: look at the contact sheet and
  `issues`, fix, repeat. The frames of that render are kept for the next step.

## 3. Compose

`compose_video(segments=["fig1", "fig2", {{"figure_id": "fig3", "hold_end": 1}}],
output="explainer.mp4", transition="fade")`

* A segment is a figure id (its last `animate`) or `{{"figure_id", "hold_end"}}`; a figure never
  animated is a still: `{{"figure_id": "fig4", "duration": 3}}`.
* `transition`: `fade` (to the page colour and back), `crossfade` (overlap), `cut`;
  `transition_s` its length. `fps` defaults to the fastest scene's.
* Scenes already rendered by `animate` are reused (`source: "cached"`); a scene changed since is
  rendered again (`"rendered"`), which takes as long as its `animate` did.
* Omit `output` to preview: the contact sheet shows the start, middle and end of every scene.
* `.mp4` for anything long (needs `ffmpeg`); a GIF is limited to {max_gif} frames. At most
  {max_frames} frames in all. `audio="voice.mp3"` adds a soundtrack to an .mp4/.webm.

## 4. Hand it over

Give the user the file, and offer `export_script(figure_id, kind="animation")` per scene for the
Python behind it.

## Pacing

1-2 s per play, a 0.5-1 s stage with nothing in it after each idea, `hold_end` on the last scene.
Keep each frame uncluttered: what is not being talked about any more fades out.
"""
