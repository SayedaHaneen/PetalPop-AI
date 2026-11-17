"""
Helper utilities for PetalPop AI:
- File validation
- Image loading & preprocessing
- Collection management (JSON persistence)
- Plot styling
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PetalPopAI")

COLLECTION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "collection.json")

def validate_image_file(uploaded_file) -> bool:
    """Validate that the uploaded file is a genuine and supported image."""
    if uploaded_file is None:
        return False
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    filename = getattr(uploaded_file, "name", "").lower()
    if not any(filename.endswith(ext) for ext in valid_extensions):
        return False
    try:
        # Check by opening with PIL
        img = Image.open(uploaded_file)
        img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image file: {e}")
        return False

def load_image_for_display(uploaded_file) -> Optional[Image.Image]:
    """Safely open PIL Image converted to RGB."""
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Error loading image for display: {e}")
        return None

def init_collection_db():
    """Ensure the collection storage JSON file exists."""
    os.makedirs(os.path.dirname(COLLECTION_FILE), exist_ok=True)
    if not os.path.exists(COLLECTION_FILE):
        with open(COLLECTION_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

def save_to_collection(
    flower_name: str,
    scientific_name: str,
    confidence: float,
    image_path: str,
    notes: str = ""
) -> Dict[str, Any]:
    """Save an identified flower to the local personal collection."""
    init_collection_db()
    entry = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "flower_name": flower_name,
        "scientific_name": scientific_name,
        "confidence": round(confidence, 2),
        "image_path": image_path,
        "notes": notes
    }
    try:
        with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
            collection = json.load(f)
        collection.insert(0, entry)  # Prepend newest
        with open(COLLECTION_FILE, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)
        return entry
    except Exception as e:
        logger.error(f"Failed to save flower to collection: {e}")
        return {}

def load_collection() -> List[Dict[str, Any]]:
    """Load all saved flowers from the personal collection."""
    init_collection_db()
    try:
        with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        return []

def delete_from_collection(entry_id: str) -> bool:
    """Delete an item from collection by its entry id."""
    init_collection_db()
    try:
        with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
            collection = json.load(f)
        new_collection = [item for item in collection if item.get("id") != entry_id]
        with open(COLLECTION_FILE, "w", encoding="utf-8") as f:
            json.dump(new_collection, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to delete entry {entry_id}: {e}")
        return False
