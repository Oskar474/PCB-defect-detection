import os
import gradio as ui
from ultralytics import YOLO

def test(img):
    return img

ui.Interface(test, ui.Image(type="numpy"), ui.Image(type="numpy")).launch()