import numpy as np
from glplot.engine import GPULinePlot
from glplot.options import EngineOptions


def smoke_test():
    print("Running smoke test (instantiation)...")
    opts = EngineOptions()
    # We won't call run() because it opens a window/context
    engine = GPULinePlot(options=opts)

    n = 100
    ab = np.random.randn(n, 2).astype(np.float32)
    engine.set_lines_ab(ab)

    print("Scene lines count:", engine.scene.lines.count)
    assert engine.scene.lines.count == n

    print("Smoke test passed: Engine instantiated and data uploaded (to scene state).")


if __name__ == "__main__":
    smoke_test()
