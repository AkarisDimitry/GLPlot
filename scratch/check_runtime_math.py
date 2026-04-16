import os
import sys

# Force current directory into path
sys.path.insert(0, os.path.abspath("."))

import glplot
from glplot.controllers import CameraController
import inspect

print("--- ENVIRONMENT INFO ---")
print(f"glplot file: {glplot.__file__}")
print(f"sys.path[0]: {sys.path[0]}")

print("\n--- RUNTIME SOURCE (world_window) ---")
source = inspect.getsource(CameraController.world_window)
print(source)

print("\n--- RUNTIME SOURCE (fit_bounds) ---")
source_fit = inspect.getsource(CameraController.fit_bounds)
print(source_fit)

print("\n--- RUNTIME EVALUATION ---")
from glplot.core.legacy import CameraState
from glplot.options import EngineOptions

cam = CameraState()
ctrl = CameraController(cam, EngineOptions())

print(f"Initial: zoom_x={cam.zoom_x}, zoom_y={cam.zoom_y}")
ctrl.fit_bounds(xmin=-2, xmax=2, ymin=-1000, ymax=1000, width=800, height=600)
print(f"After fit_bounds(-2, 2, -1000, 1000): zoom_x={cam.zoom_x}, zoom_y={cam.zoom_y}")

l, r, b, t = ctrl.world_window(800, 600)
print(f"world_window(800, 600) output: ({l}, {r}, {b}, {t})")

if abs(l - (-2.0)) < 1e-5:
    print("\nRESULT: SUCCESS - Source is producing correct decoupled values.")
else:
    print(f"\nRESULT: FAILURE - Source is producing coupled values. (Expected l=-2.0, got l={l})")
