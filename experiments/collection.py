
import pandas as pd

import shutil
import random
from pathlib import Path
import json
import os
import shutil

# ---------------------------------------

def sample_files_in_dir(dir_path: str, n: int) -> list[Path]:
    # Ensure we are working with an absolute, normalized path
    base = Path(dir_path).expanduser().resolve()

    # Collect **only files** (skip sub‑directories, symlinks, etc.)
    all_files = [p for p in base.rglob("*") if p.is_file()]

    # If the directory is empty, return an empty list
    if not all_files:
        return []

    # Limit *n* to the number of available files
    n = min(n, len(all_files))

    # Randomly choose without replacement
    chosen = random.sample(all_files, k=n)

    # Paths are already absolute because we started with a resolved base
    return chosen

def filename_and_ext(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    return p.stem + p.suffix

def copy_file(src: str | Path, dst: str | Path) -> Path:
    src_path = Path(src).expanduser().resolve()
    dst_path = Path(dst).expanduser()

    # If dst is a directory (existing, or marked with a trailing separator),
    # place the file inside it using the source filename.
    if dst_path.is_dir() or str(dst).endswith(("/", "\\", os.sep)):
        dst_path.mkdir(parents=True, exist_ok=True)
        dst_path = dst_path / src_path.name

    # Ensure the parent directory of the final destination exists.
    dst_path = dst_path.resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_path, dst_path)
    return dst_path

# ---------------------------------------

with open("./imagenet-labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

for model in ["vit_b_16", "resnet50"]:
    df = pd.read_csv(f'./results/negative-similarity/{model}/all-minmax-global-unstructured-L1-0.csv')
    cf = df[df['least_class_count'] > 15]

    N = 5
    for _, row in cf.iterrows():
        tar_cls = row["target_class_idx"].item()
        neg_cls = row["least_class_idx"].item()
        tar_cls_s = f'{tar_cls:05d}'
        neg_cls_s = f'{neg_cls:05d}'
        neg_count = row["least_class_count"].item()

        tar_samples_paths = sample_files_in_dir(f"D:/sautkin.imagenet1kvalid/{tar_cls_s}", N)
        neg_samples_paths = sample_files_in_dir(f"D:/sautkin.imagenet1kvalid/{neg_cls_s}", N)

        print(tar_cls_s, neg_cls_s)
        for i, t, n in zip(range(N), tar_samples_paths, neg_samples_paths):
            dir = f"./dist/{model}/{tar_cls_s}-{neg_cls_s}-({neg_count})+{labels[str(tar_cls)]}-{labels[str(neg_cls)]}"
            copy_file(t, f'{dir}/{tar_cls_s}-pos-{i}-{filename_and_ext(t)}')
            copy_file(n, f'{dir}/{neg_cls_s}-neg-{i}-{filename_and_ext(n)}')


