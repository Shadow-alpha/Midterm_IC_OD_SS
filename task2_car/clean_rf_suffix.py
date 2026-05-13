import argparse
import os


def build_target_name(filename, marker=".rf."):
    stem, ext = os.path.splitext(filename)
    if marker not in stem:
        return None
    prefix = stem.split(marker, 1)[0]
    return f"{prefix}{ext.lower()}"


def resolve_collision(folder, name):
    base, ext = os.path.splitext(name)
    index = 1
    candidate = name
    while os.path.exists(os.path.join(folder, candidate)):
        candidate = f"{base}_{index}{ext}"
        index += 1
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory to scan")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".txt"],
        help="Extensions to rename",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print changes")
    args = parser.parse_args()

    exts = {ext.lower() for ext in args.exts}

    renamed = 0
    for dirpath, _, files in os.walk(args.root):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() not in exts:
                continue

            target = build_target_name(name)
            if not target or target == name:
                continue

            src = os.path.join(dirpath, name)
            final_name = resolve_collision(dirpath, target)
            dst = os.path.join(dirpath, final_name)

            if args.dry_run:
                print(f"DRY-RUN: {src} -> {dst}")
            else:
                os.rename(src, dst)
                print(f"RENAMED: {src} -> {dst}")
                renamed += 1

    if not args.dry_run:
        print(f"Done. Renamed {renamed} file(s).")


if __name__ == "__main__":
    main()

# python clean_rf_suffix.py --root /home/zhangtianyi/homework/Midterm_IC_OD_SS/task2/trafic_data --dry-run
# python clean_rf_suffix.py --root /home/zhangtianyi/homework/Midterm_IC_OD_SS/task2/trafic_data