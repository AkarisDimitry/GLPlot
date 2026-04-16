import glplot
import os
print(f"glplot file: {glplot.__file__}")
print(f"glplot location: {os.path.dirname(glplot.__file__)}")

from glplot.controllers import CameraController
import inspect
print(f"CameraController source file: {inspect.getfile(CameraController)}")
