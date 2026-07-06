import numpy as np

import glplot.pyplot as gplt

# Simple sequence example
x = np.linspace(0, 10, 100)
y = np.sin(x)

gplt.figure("Simple Plot Demo")

# Activate adaptive subsampling for massive plots in standard sequences
gplt.set_lod(enabled=True, max_lines_per_px=200)


for n in range(10000):
    gplt.plot(x, y + float(n) ** 2 / 100000.0, color=(1.0, 0.0, 0.0, 1.0))  # Red line
    gplt.scatter(
        x, y + float(n) ** 2 / 100000.0, color=(0.0, 0.0, 1.0, 1.0), size=12.0
    )  # Blue points

gplt.set_global_alpha(1.0)
gplt.show()
