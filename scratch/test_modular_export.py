import os

import numpy as np

import glplot.pyplot as gplt


def test_export():
    print("Testing modularized export...")
    n = 100_000
    a = np.random.randn(n)
    b = np.random.randn(n)

    gplt.figure("Test Export", hud=True, budget=5)
    gplt.lines(a, b, x_range=(-5, 5), color="blue")

    # Test profile
    gplt.profile("quality")

    # Test export
    out_file = "test_export_highres.png"
    if os.path.exists(out_file):
        os.remove(out_file)

    gplt.savefig(out_file, scale=2.5)

    if os.path.exists(out_file):
        print(f"Success: Exported {out_file}")
        # size check would be good but requires PIL or similar
    else:
        print("Failure: Export file not found")


if __name__ == "__main__":
    test_export()
