import os
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def line_side(line_start, line_end, point):
    x1, y1 = line_start
    x2, y2 = line_end
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def center_from_xyxy(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
