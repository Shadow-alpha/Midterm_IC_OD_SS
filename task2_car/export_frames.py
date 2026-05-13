import argparse
import os
import cv2
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    frame_set = set(args.frames)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    index = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index in frame_set:
            out_path = os.path.join(args.out_dir, f"frame_{index:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        index += 1

    cap.release()
    print(f"Saved {saved} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
