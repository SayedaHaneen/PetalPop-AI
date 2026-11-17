"""
Explainable AI Module: Gradient-weighted Class Activation Mapping (Grad-CAM)
for MobileNetV2 in PetalPop AI.
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def find_target_conv_layer(model: tf.keras.Model) -> str:
    """
    Dynamically inspects the model or base MobileNetV2 to find the last 4D conv layer.
    """
    # First search in base model if nested
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            for sub_layer in reversed(layer.layers):
                if len(getattr(sub_layer, "output_shape", ())) == 4 or isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.ReLU)):
                    return f"{layer.name}:{sub_layer.name}"
        if len(getattr(layer, "output_shape", ())) == 4 or isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    # Default fallback for MobileNetV2
    return "out_relu"

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = None,
    pred_index: int = None
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for a given input image array and target class prediction index.
    """
    # Identify base model and conv layer
    base_model = None
    target_sub_layer_name = "out_relu"

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        # Check for out_relu or Conv_1
        candidate_names = ["out_relu", "Conv_1", "conv2d"]
        for cand in candidate_names:
            try:
                base_model.get_layer(cand)
                target_sub_layer_name = cand
                break
            except Exception:
                continue

        # Create gradient model using base_model directly
        grad_model = tf.keras.models.Model(
            inputs=[base_model.input],
            outputs=[base_model.get_layer(target_sub_layer_name).output, base_model.output]
        )

        # Retrieve layers after base model
        head_layers = []
        found_base = False
        for layer in model.layers:
            if found_base:
                head_layers.append(layer)
            if layer == base_model:
                found_base = True

        # Preprocess input matching model input layer
        preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        with tf.GradientTape() as tape:
            conv_outputs, base_features = grad_model(preprocessed_img)
            tape.watch(conv_outputs)

            x = base_features
            for head_layer in head_layers:
                x = head_layer(x)
            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradients of the predicted class score with respect to feature map outputs
        grads = tape.gradient(class_channel, conv_outputs)
        # Global average pooling of gradients: mean intensity of each feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature map channels by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to retain only features with positive influence
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()

    else:
        # Fallback if unnested
        return np.ones((7, 7), dtype=np.float32)

def generate_gradcam_overlay(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlays Grad-CAM heatmap on original PIL image.
    Returns:
    - colored_heatmap (RGB)
    - superimposed_image (RGB)
    """
    img_rgb = np.array(pil_image)
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    
    # Scale to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colorize heatmap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose on original image
    superimposed = np.float32(colored_heatmap) * alpha + np.float32(img_rgb) * (1.0 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return colored_heatmap, superimposed
