import numpy as np
import glplot.pyplot as gplt

def demo_density_gain():
    n = 1_000_000
    a = np.random.randn(n) * 0.1
    b = np.random.randn(n) * 0.1
    
    # Pre-set some gain
    gplt.density_gain(15.0)
    gplt.figure("Density Gain Test", density=True, hud=True)
    gplt.lines(a, b, x_range=(-1, 1), color='yellow')
    
    print("Check if the 'Density Factor' slider appears in the HUD and affects the heatmap.")
    gplt.show()

if __name__ == "__main__":
    demo_density_gain()
