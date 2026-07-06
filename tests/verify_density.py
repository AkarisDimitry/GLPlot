import os

import numpy as np

import glplot.pyplot as plt

# Generate some data
n = 1_000_000
a = np.random.normal(0, 0.4, n)
b = np.random.normal(0, 1.0, n)

# Plot
plt.figure("Density Verification")
plt.lines(a, b, x_range=(-4, 4), color="blue")
plt.title("Density Heatmap")

# Save with density=True
output = "density_test.png"
if os.path.exists(output):
    os.remove(output)

print("Running savefig(density=True)...")
plt.savefig(output, density=True)

if os.path.exists(output):
    print(f"Success! {output} created.")
else:
    print("Failed: output file not created.")
