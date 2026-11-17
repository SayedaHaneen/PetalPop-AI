"""
Data acquisition, organization, preprocessing, and augmentation pipeline
for Oxford 102 Flowers dataset.
"""

import os
import tarfile
import urllib.request
import logging
import shutil
import numpy as np
import scipy.io
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataPreprocessing")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

URL_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
URL_LABELS = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
URL_SPLITS = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

# Alternative reliable mirrors
MIRROR_IMAGES = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz"

def download_file(url: str, dest_path: str, retries: int = 2) -> bool:
    """Download a file with progress logging and timeout."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        logger.info(f"File already exists at {dest_path}")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading from {url} to {dest_path}...")
    
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PetalPopAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            logger.info(f"Download complete: {dest_path} ({os.path.getsize(dest_path)} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
    return False

def acquire_oxford_102(raw_dir: str = RAW_DIR) -> bool:
    """
    Downloads images and matlab metadata files for Oxford 102 Flowers dataset.
    """
    os.makedirs(raw_dir, exist_ok=True)
    
    images_tar = os.path.join(raw_dir, "102flowers.tgz")
    labels_mat = os.path.join(raw_dir, "imagelabels.mat")
    setid_mat = os.path.join(raw_dir, "setid.mat")

    # Download labels and setid
    labels_ok = download_file(URL_LABELS, labels_mat)
    setid_ok = download_file(URL_SPLITS, setid_mat)
    
    # Download images tar
    img_ok = download_file(URL_IMAGES, images_tar)
    if not img_ok:
        logger.info("Attempting backup mirror for images...")
        img_ok = download_file(MIRROR_IMAGES, images_tar)

    return (labels_ok and setid_ok and img_ok)

def extract_and_organize_dataset(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
):
    """
    Extracts 102flowers.tgz and organizes images into:
    processed/train/{class_id}/
    processed/val/{class_id}/
    processed/test/{class_id}/
    Guarantees strict separation to eliminate data leakage.
    """
    images_tar = os.path.join(raw_dir, "102flowers.tgz")
    labels_mat_path = os.path.join(raw_dir, "imagelabels.mat")
    
    extract_target = os.path.join(raw_dir, "jpg")
    if not os.path.exists(extract_target) or len(os.listdir(extract_target)) < 100:
        logger.info(f"Extracting {images_tar}...")
        with tarfile.open(images_tar, "r:gz") as tar:
            tar.extractall(path=raw_dir)
        logger.info("Extraction complete.")

    # Load labels
    mat = scipy.io.loadmat(labels_mat_path)
    labels = mat["labels"][0]  # 1-indexed (1 to 102)

    image_filenames = sorted(os.listdir(extract_target))
    # Filter only jpg files
    image_filenames = [f for f in image_filenames if f.lower().endswith(".jpg")]
    
    assert len(image_filenames) == len(labels), f"Images ({len(image_filenames)}) vs Labels ({len(labels)}) mismatch!"

    # Zero-indexed classes for standard neural net training
    zero_indexed_labels = labels - 1

    # Stratified split: Train (70%), Val (15%), Test (15%)
    indices = np.arange(len(image_filenames))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, zero_indexed_labels, test_size=(val_size + test_size),
        stratify=zero_indexed_labels, random_state=random_state
    )
    val_ratio_in_temp = val_size / (val_size + test_size)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=(1.0 - val_ratio_in_temp),
        stratify=y_temp, random_state=random_state
    )

    logger.info(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    splits = {
        "train": (train_idx, y_train),
        "val": (val_idx, y_val),
        "test": (test_idx, y_test)
    }

    # Copy / organize files
    for split_name, (split_indices, split_labels) in splits.items():
        split_dir = os.path.join(processed_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for idx, class_id in zip(split_indices, split_labels):
            src_file = os.path.join(extract_target, image_filenames[idx])
            class_folder = os.path.join(split_dir, f"{class_id:03d}")
            os.makedirs(class_folder, exist_ok=True)
            dst_file = os.path.join(class_folder, image_filenames[idx])
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

    logger.info(f"Dataset successfully organized in {processed_dir}")

def get_data_augmentation_layers():
    """
    Keras Sequential data augmentation pipeline:
    - Random Flip (horizontal & vertical)
    - Random Rotation (-15 to +15 deg)
    - Random Zoom (up to 15%)
    - Random Contrast (up to 15%)
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.12),
    ], name="data_augmentation")

def build_datasets(
    processed_dir: str = PROCESSED_DIR,
    image_size: tuple = (224, 224),
    batch_size: int = 32
):
    """
    Loads train, validation, and test tf.data.Dataset from organized folders.
    """
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Apply data augmentation only to the training dataset
    data_aug = get_data_augmentation_layers()
    train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetching for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
