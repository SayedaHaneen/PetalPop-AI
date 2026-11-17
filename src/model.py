"""
Model architecture module for PetalPop AI:
- MobileNetV2 Transfer Learning Architecture
- Feature extraction head
- Fine-tuning configuration
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_transfer_learning_model(
    input_shape=(224, 224, 3),
    num_classes=102,
    dropout_rate=0.35,
    dense_units=256
) -> tf.keras.Model:
    """
    Builds the PetalPop MobileNetV2 transfer learning model.
    Base model initialized with ImageNet weights and frozen initially.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Freeze base model initially
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # Preprocessing layer for MobileNetV2 (-1 to 1 scaling)
    # Using keras MobileNetV2 preprocess_input formulation
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Feature extraction via base model
    x = base_model(x, training=False)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate / 2.0, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PetalPop_MobileNetV2")
    return model

def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    unfreeze_from_layer: int = 100,
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    """
    Unfreezes the upper layers of the base MobileNetV2 model for fine-tuning.
    Compiles with a lower learning rate.
    """
    # Locate the base MobileNetV2 layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not locate base MobileNetV2 model inside wrapper.")

    base_model.trainable = True
    
    # Freeze all layers up to `unfreeze_from_layer`
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from_layer:]:
        # Keep BatchNormalization in inference mode during fine-tuning for stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
