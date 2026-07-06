import os

import numpy as np

import glplot.pyplot as plt

# Generate some data
n = 1000
a = np.random.randn(n)
b = np.random.randn(n)

# Plot
plt.figure("Headless Test")
plt.lines(a, b, x_range=(-5, 5), color="blue")
plt.title("Headless Export")

# Save
output = "test_output.png"
if os.path.exists(output):
    os.remove(output)

print("Running savefig...")
plt.savefig(output)

if os.path.exists(output):
    print(f"Success! {output} created.")
else:
    print("Failed: output file not created.")
