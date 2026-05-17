import argparse
import os
import cv2
from ultralytics import YOLO
from utils import load_config, ensure_dir, line_side, center_from_xyxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    track_cfg = cfg["track"]
    count_cfg = cfg["counting"]
    analysis_cfg = cfg.get("analysis") or {}

    ensure_dir(track_cfg["output_dir"])
    frame_dir = os.path.join(track_cfg["output_dir"], "frames")
    if analysis_cfg.get("save_frames", True):
        ensure_dir(frame_dir)

    line_start = tuple(int(v) for v in count_cfg["line_start"])
    line_end = tuple(int(v) for v in count_cfg["line_end"])

    model = YOLO(track_cfg["model_path"])
    track_kwargs = {
        "source": track_cfg["video_path"],
        "conf": track_cfg["conf"],
        "iou": track_cfg["iou"],
        "imgsz": track_cfg["imgsz"],
        "tracker": track_cfg["tracker"],
        "persist": True,
        "stream": True,
        "show": track_cfg.get("show", False),
    }
    device = track_cfg.get("device")
    if device is not None:
        track_kwargs["device"] = device
    results = model.track(**track_kwargs)

    cap = cv2.VideoCapture(track_cfg["video_path"])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    out_path = os.path.join(track_cfg["output_dir"], "tracked_counted.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    last_side = {}
    counted_ids = set()
    total_count = 0
    frame_index = 0
    debug_frames = set(analysis_cfg.get("debug_frames", []))
    save_all_frames = analysis_cfg.get("save_all_frames", False)

    for res in results:
        frame = res.plot()
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.tolist()
            boxes = res.boxes.xyxy.tolist()

            for track_id, box in zip(ids, boxes):
                cx, cy = center_from_xyxy(box)
                current_side = line_side(line_start, line_end, (cx, cy))
                prev_side = last_side.get(track_id)
                if prev_side is not None:
                    crossed = (current_side > 0 > prev_side) or (current_side < 0 < prev_side)
                    if crossed and track_id not in counted_ids:
                        total_count += 1
                        counted_ids.add(track_id)
                last_side[track_id] = current_side

        cv2.line(
            frame,
            line_start,
            line_end,
            tuple(count_cfg["line_color"]),
            count_cfg["line_thickness"],
        )
        cv2.putText(
            frame,
            f"Count: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            count_cfg["font_scale"],
            tuple(count_cfg["text_color"]),
            2,
        )

        writer.write(frame)

        if analysis_cfg.get("save_frames", True) and (save_all_frames or frame_index in debug_frames):
            out_frame = os.path.join(frame_dir, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(out_frame, frame)

        frame_index += 1

    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
