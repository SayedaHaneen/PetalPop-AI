"""
Robust download and preparation of Oxford 102 dataset.
Streams download with live progress and extracts into data/raw/jpg.
Organizes into train, val, and test splits with zero data leakage.
"""

import os
import sys
import time
import tarfile
import shutil
import scipy.io
import numpy as np
import requests
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. Download imagelabels.mat
labels_path = os.path.join(RAW_DIR, "imagelabels.mat")
if not os.path.exists(labels_path):
    print("Downloading imagelabels.mat...", flush=True)
    r = requests.get("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", timeout=30)
    with open(labels_path, "wb") as f:
        f.write(r.content)
    print("Downloaded imagelabels.mat", flush=True)

# 2. Download setid.mat
setid_path = os.path.join(RAW_DIR, "setid.mat")
if not os.path.exists(setid_path):
    print("Downloading setid.mat...", flush=True)
    r = requests.get("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", timeout=30)
    with open(setid_path, "wb") as f:
        f.write(r.content)
    print("Downloaded setid.mat", flush=True)

# 3. Stream download 102flowers.tgz
tar_path = os.path.join(RAW_DIR, "102flowers.tgz")
if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 300 * 1024 * 1024:
    url = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz"
    print(f"Downloading {url} to {tar_path}...", flush=True)
    r = requests.get(url, stream=True, timeout=30)
    total_length = int(r.headers.get("content-length", 0))
    dl = 0
    t0 = time.time()
    last_print = t0
    
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024 * 2): # 2MB chunks
            if chunk:
                f.write(chunk)
                dl += len(chunk)
                now = time.time()
                if now - last_print > 3: # Print every 3s
                    percent = (dl / total_length) * 100 if total_length else 0
                    mb = dl / (1024 * 1024)
                    speed = mb / (now - t0 + 1e-5)
                    print(f"Downloaded: {mb:.1f} MB ({percent:.1f}%) | Speed: {speed:.1f} MB/s", flush=True)
                    last_print = now

    print(f"Download complete: {tar_path} ({os.path.getsize(tar_path) / (1024*1024):.1f} MB)", flush=True)

# 4. Extract
jpg_dir = os.path.join(RAW_DIR, "jpg")
if not os.path.exists(jpg_dir) or len(os.listdir(jpg_dir)) < 1000:
    print(f"Extracting {tar_path}...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)
    
    # Check if extracted inside oxford-102-flowers/
    sub = os.path.join(RAW_DIR, "oxford-102-flowers")
    if os.path.exists(sub):
        if not os.path.exists(jpg_dir):
            sub_jpg = os.path.join(sub, "jpg")
            if os.path.exists(sub_jpg):
                shutil.move(sub_jpg, jpg_dir)
            else:
                shutil.move(sub, jpg_dir)
        shutil.rmtree(sub, ignore_errors=True)

    # Search for folder with images
    if not os.path.exists(jpg_dir):
        for root, dirs, files in os.walk(RAW_DIR):
            jpgs = [f for f in files if f.lower().endswith(".jpg")]
            if len(jpgs) > 1000:
                print(f"Found {len(jpgs)} images in {root}", flush=True)
                if root != jpg_dir:
                    os.rename(root, jpg_dir)
                break

print("Extraction verified. Images in jpg/:", len(os.listdir(jpg_dir)) if os.path.exists(jpg_dir) else 0, flush=True)

# 5. Organize into train, val, test
print("Organizing into train/val/test splits...", flush=True)
mat = scipy.io.loadmat(labels_path)
labels = mat["labels"][0] # 1 to 102

image_files = sorted([f for f in os.listdir(jpg_dir) if f.lower().endswith(".jpg")])
print(f"Total image files found: {len(image_files)}, labels: {len(labels)}", flush=True)

assert len(image_files) == len(labels), f"Images ({len(image_files)}) vs Labels ({len(labels)}) mismatch!"

zero_indexed_labels = labels - 1

# Stratified split: Train (70%), Val (15%), Test (15%)
indices = np.arange(len(image_files))
train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices, zero_indexed_labels, test_size=0.30,
    stratify=zero_indexed_labels, random_state=42
)
val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx, y_temp, test_size=0.50,
    stratify=y_temp, random_state=42
)

splits = {
    "train": (train_idx, y_train),
    "val": (val_idx, y_val),
    "test": (test_idx, y_test)
}

for split_name, (split_indices, split_labels) in splits.items():
    split_dir = os.path.join(PROCESSED_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    count = 0
    for idx, class_id in zip(split_indices, split_labels):
        src_file = os.path.join(jpg_dir, image_files[idx])
        class_folder = os.path.join(split_dir, f"{class_id:03d}")
        os.makedirs(class_folder, exist_ok=True)
        dst_file = os.path.join(class_folder, image_files[idx])
        if not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            count += 1
    print(f"Split '{split_name}': {len(split_indices)} images organized.", flush=True)

print("Dataset preparation complete!", flush=True)
