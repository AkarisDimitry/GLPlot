"""Where does the sine wave come from? -- a step-by-step explainer in the 3Blue1Brown style.

``plt.plot_style("3b1b")`` gives the look of 3Blue1Brown's videos, which are made with manim: a
black page, manim's BLUE_C / YELLOW_C / TEAL palette, white serif text with ``$...$`` maths in
Computer Modern, and (in the exported GIF/MP4) axes that are two lines crossing at the origin with
arrow tips instead of a box.

The video is built the way a manim scene is, one *play* at a time, each eased with manim's own
``smooth`` rate function:

1. the unit circle is drawn in (manim's ``Create``);
2. the question is written letter by letter (``Write``);
3. a point goes once round the circle; its height is carried across by a dashed line and traces
   ``sin(theta)`` on the right, while the formula is written and a live readout follows the angle;
4. the formula is pointed at -- it turns yellow and grows, then settles (``Indicate``);
5. the last picture holds.

Every frame redraws the scene from one number per element -- how far through its play it is -- so
there is no state to drift: ``update(i)`` is a pure function of the frame index, which is what
makes the saved GIF identical to what ``plt.show()`` plays.
"""

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

FPS = 20
WHITE, BLUE, YELLOW, TEAL = "#FFFFFF", "#58C4DD", "#F7D96F", "#5CD0B3"

# -- the plays, in seconds: (start, end) --------------------------------------------------
CIRCLE = (0.0, 1.2)
QUESTION = (1.0, 2.4)
TRACE = (2.6, 7.6)
FORMULA = (2.6, 3.6)
INDICATE = (7.9, 8.8)
END = 10.0
FRAMES = int(END * FPS)

CX = -1.6  # the circle's centre on the x axis; the sine is drawn from x = 0


def smooth(t):
    """manim's default rate function: 0 -> 1 with zero speed and acceleration at both ends."""
    t = np.clip(t, 0.0, 1.0)
    s = 1.0 - t
    return t**3 * (10.0 * s * s + 5.0 * s * t + t * t)


def progress(time, span):
    """How far through a play (``span`` = start, end in seconds) ``time`` is, eased: 0..1."""
    start, end = span
    return float(smooth((time - start) / (end - start)))


def there_and_back(t):
    """0 -> 1 -> 0, smoothly: the shape of manim's ``Indicate``."""
    return float(smooth(1.0 - abs(2.0 * np.clip(t, 0.0, 1.0) - 1.0)))


def written(text, share):
    """The first ``share`` of ``text``, a ``$...$`` formula counting as a single piece."""
    import re

    pieces = re.findall(r"\$[^$]*\$|.", text)
    return "".join(pieces[: int(np.ceil(share * len(pieces) - 1e-9))])


def readout(theta, height):
    """The live numbers under the graph (``+0.00`` rather than ``-0.00`` at the crossings)."""
    sine = f"{height:+.2f}".replace("-0.00", "+0.00")
    return rf"$\theta = {np.degrees(theta):5.1f}^\circ \qquad \sin\theta = {sine}$"


QUESTION_TEXT = "Where does the sine wave come from?"
theta_full = np.linspace(0.0, 2.0 * np.pi, 400)

# No window name: figure("...") would also become the plot title, and the question is the title.
fig = plt.figure(figsize=(8.0, 4.5))
plt.plot_style("3b1b")


def update(frame: int):
    time = frame / FPS
    plt.cla()

    # 1. the circle, drawn in from angle 0
    drawn = progress(time, CIRCLE)
    if drawn > 0:
        arc = theta_full[: max(2, int(drawn * len(theta_full)))]
        plt.plot(CX + np.cos(arc), np.sin(arc), color=BLUE, linewidth=3)

    # 3. the point going round, its radius, its height carried across, the sine it traces
    turn = progress(time, TRACE)
    if time >= TRACE[0]:
        theta = 2.0 * np.pi * turn
        px, py = CX + np.cos(theta), np.sin(theta)
        plt.plot([CX, px], [0.0, py], color=WHITE, linewidth=2)
        plt.plot([px, theta], [py, py], color=YELLOW, linewidth=1.5, linestyle="--")
        traced = theta_full[theta_full <= theta + 1e-9]
        if len(traced) > 1:
            plt.plot(traced, np.sin(traced), color=TEAL, linewidth=3.5)
        plt.scatter([px], [py], c=[WHITE], size=14)
        plt.scatter([theta], [py], c=[YELLOW], size=14)
        plt.text(
            2.3,
            -1.85,
            readout(theta, py),
            fontsize=18,
            color=(1, 1, 1, progress(time, FORMULA)),
        )

    # 2. the question, written letter by letter
    share = progress(time, QUESTION)
    if share > 0:
        # a box in the page colour behind it hides the y axis where the text crosses it
        # (manim's BackgroundRectangle)
        # layout_text: placed where its letters sit in the whole question, so it is written
        # in place rather than re-centred as it grows
        plt.text(2.0, 2.05, written(QUESTION_TEXT, share), fontsize=24, ha="center",
                 layout_text=QUESTION_TEXT,
                 bbox=dict(facecolor="black", edgecolor="none", pad=4))  # fmt: skip

    # 3/4. the formula: written, then pointed at (yellow and a little bigger, then back)
    share = progress(time, FORMULA)
    if share > 0:
        pulse = there_and_back((time - INDICATE[0]) / (INDICATE[1] - INDICATE[0]))
        mix = np.array([1.0, 1.0, 1.0]) * (1 - pulse) + np.array([0.969, 0.851, 0.435]) * pulse
        plt.text(
            CX,
            -1.85,
            written(r"$y = \sin\theta$", share),
            layout_text=r"$y = \sin\theta$",
            fontsize=26 * (1 + 0.25 * pulse),
            color=tuple(mix),
            ha="center",
        )

    plt.xlim(-2.9, 6.9)
    plt.ylim(-2.3, 2.5)
    plt.gca().set_aspect("equal")
    plt.xticks(
        [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    plt.yticks([-1, 1])
    plt.xlabel(r"$\theta$")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=1000 / FPS)
# plt.show()   # play it in a window instead
ani.save("examples/gallery/animations/results/18_3b1b_unit_circle_sine.gif", fps=FPS)
