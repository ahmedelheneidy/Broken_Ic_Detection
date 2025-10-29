import torch
import os

from IPython.display import Image, clear_output  # to display images

print(
    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")
os.environ["DATASET_DIRECTORY"] = "/custom/datasets"

from roboflow import Roboflow
rf = Roboflow(api_key="9Dh2krsdyUXB6OyqmJpF")
project = rf.workspace("mvi-owp6u").project("task-2-nnqlo")
dataset = project.version(3).download("yolov5")