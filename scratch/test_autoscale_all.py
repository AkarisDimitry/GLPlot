import numpy as np
import glplot.pyplot as gplt


def test_multi_layer_autoscale():
    print("Testing multi-layer autoscale...")

    gplt.figure("Multi-Layer Test", width=800, height=600)

    # Layer 1: centered at (0, 0)
    gplt.plot([-1, 1], [-1, 1], color="blue", label="Diag 1")

    # Layer 2: far away at (10, 10)
    gplt.plot([9, 11], [9, 11], color="red", label="Diag 2")

    # Autoscale
    gplt.autoscale(tight=True)

    lx = gplt.xlim()
    ly = gplt.ylim()

    print(f"X limits: {lx}")
    print(f"Y limits: {ly}")

    # Expected: Union of [-1, 1] and [9, 11] => [-1, 11]
    expected = (-1.0, 11.0)

    assert abs(lx[0] - expected[0]) < 1e-5
    assert abs(lx[1] - expected[1]) < 1e-5
    assert abs(ly[0] - expected[0]) < 1e-5
    assert abs(ly[1] - expected[1]) < 1e-5

    print("Success: Autoscale considers all data.")


if __name__ == "__main__":
    test_multi_layer_autoscale()
