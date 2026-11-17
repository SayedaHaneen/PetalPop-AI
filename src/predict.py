"""
Inference & Prediction Engine for PetalPop AI:
- Robust image validation and loading
- MobileNetV2-compatible preprocessing
- Top-K class probability computation
- Uncertainty estimation
- Integration with Flower Information database and Grad-CAM
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

from src.flower_info import get_flower_by_zero_index
from src.gradcam import make_gradcam_heatmap, generate_gradcam_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PetalPopPredictor")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.keras")

class FlowerClassifier:
    """Predictor class managing model loading, preprocessing, and inference."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[tf.keras.Model] = None
        self.load_model()

    def load_model(self) -> bool:
        """Load the trained Keras model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Please train or download model.")
            return False
        try:
            logger.info(f"Loading trained model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False

    def is_ready(self) -> bool:
        return self.model is not None

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Resizes and formats image for MobileNetV2.
        Returns:
            np.ndarray of shape (1, 224, 224, 3) with float32 values.
        """
        img = pil_image.convert("RGB")
        img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # Shape (1, 224, 224, 3)
        return arr

    def predict(
        self,
        pil_image: Image.Image,
        top_k: int = 3,
        uncertainty_threshold: float = 40.0
    ) -> Dict[str, Any]:
        """
        Runs inference on uploaded PIL image.
        Returns full prediction result dictionary.
        """
        if self.model is None:
            # Try reloading if available
            if not self.load_model():
                raise RuntimeError("PetalPop model is not loaded.")

        # Preprocess
        input_tensor = self.preprocess_image(pil_image)

        # Forward pass
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        
        # Sort indices by descending probability
        top_indices = np.argsort(predictions)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            prob_percent = float(predictions[idx]) * 100.0
            flower_info = get_flower_by_zero_index(int(idx))
            name = flower_info["common_name"] if flower_info else f"Species #{idx+1}"
            sci_name = flower_info["scientific_name"] if flower_info else "Unknown"

            results.append({
                "rank": rank,
                "class_index": int(idx),
                "flower_id": int(idx) + 1,
                "common_name": name,
                "scientific_name": sci_name,
                "confidence_percent": round(prob_percent, 2),
                "details": flower_info
            })

        top_match = results[0]
        is_uncertain = top_match["confidence_percent"] < uncertainty_threshold

        return {
            "top_match": top_match,
            "top_k": results,
            "all_probabilities": predictions,
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": uncertainty_threshold
        }

    def explain_prediction(
        self,
        pil_image: Image.Image,
        pred_index: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates Grad-CAM heatmap and overlay visualization.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        
        input_tensor = self.preprocess_image(pil_image)
        heatmap = make_gradcam_heatmap(input_tensor, self.model, pred_index=pred_index)
        colored_heatmap, overlay = generate_gradcam_overlay(pil_image, heatmap)
        return colored_heatmap, overlay
