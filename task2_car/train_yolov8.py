import argparse
import os
from ultralytics import YOLO
from utils import load_config, ensure_dir


def build_data_yaml(cfg, out_dir):
    dataset_dir = cfg["dataset"]["dataset_dir"]
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    data_yaml = os.path.join(out_dir, "data.yaml")
    content = {
        "path": dataset_dir,
        "train": cfg["dataset"]["train_images"],
        "val": cfg["dataset"]["val_images"],
        "names": cfg["dataset"]["class_names"],
    }
    with open(data_yaml, "w", encoding="utf-8") as file:
        for key, value in content.items():
            if key == "names":
                file.write("names:\n")
                for index, name in enumerate(value):
                    file.write(f"  {index}: {name}\n")
            else:
                file.write(f"{key}: {value}\n")
    return data_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg["train"]["enabled"]:
        print("Training disabled in config.")
        return

    train_cfg = cfg["train"]
    out_dir = train_cfg["output_dir"]
    ensure_dir(out_dir)
    data_yaml = build_data_yaml(cfg, out_dir)

    model = YOLO(train_cfg["model"])
    train_kwargs = {
        "data": data_yaml,
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch"],
        "imgsz": train_cfg["img_size"],
        "lr0": train_cfg["lr0"],
        "weight_decay": train_cfg["weight_decay"],
        "project": out_dir,
        "name": "",
    }
    device = train_cfg.get("device")
    if device is not None:
        train_kwargs["device"] = device
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
