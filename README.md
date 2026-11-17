# PetalPop AI: Intelligent Flower Recognition and Care Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-4B8BBE?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.15+-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SayedaHaneen/PetalPop-AI/blob/main/PetalPop_AI.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-Oxford%20102%20Flowers-green?style=flat-square)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

**A University-Level Deep Learning and Explainable Computer Vision System**  
*Fine-grained categorization of 102 flower species via MobileNetV2 Transfer Learning, Grad-CAM interpretability, and evidence-based botanical care recommendations.*

---

## 1. Project Overview

PetalPop AI is an end-to-end deep learning system designed for automated botanical recognition and plant care guidance. Fine-grained visual categorization (FGVC) of flora presents significant technical challenges due to high intra-class variance across blooming stages and subtle inter-class differences between related botanical taxa.

PetalPop AI addresses this challenge using **Transfer Learning and Fine-Tuning with MobileNetV2**, achieving high classification accuracy while maintaining lightweight, edge-compatible computational efficiency. To ensure transparency, the system implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually localize the anatomical regions (corolla, petals, stamens) that drive model predictions.

### Key Capabilities:
- **Image Identification:** Classification of flower photographs across 102 categories (JPG, JPEG, PNG).
- **Softmax Probability Scoring:** Displays top prediction confidence and ranked top-3 candidate species.
- **Uncertainty Alert:** Warns users when image ambiguity or low resolution yields sub-threshold confidence.
- **Explainable AI (Grad-CAM):** Computes class activation heatmaps overlaid directly on the input image.
- **Botanical Care Assistant:** Authoritative watering, sunlight, soil, and cultivation directives for all 102 classes.
- **Local Specimen Collection:** Persistent local collection tracking with custom user field notes.
- **Model Evaluation Dashboard:** Real-time presentation of empirical training curves, confusion matrix, and multi-class precision, recall, and F1 metrics.

---

## 2. System Architecture

```
PetalPop_AI/
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Pinned package dependencies
├── README.md                      # Academic documentation & setup manual
├── UNIVERSITY_PROJECT_REPORT.md   # Comprehensive university research report
├── download_and_prep.py           # Automated dataset acquisition & splitting
├── push_to_github.py              # Zero-git direct GitHub synchronizer
│
├── models/
│   └── best_model.keras           # Trained MobileNetV2 weights (27.7 MB, 88.04% Test Acc)
│
├── src/
│   ├── data_preprocessing.py      # Stratified splitting & data augmentation
│   ├── model.py                   # MobileNetV2 Transfer Learning & Fine-Tuning architecture
│   ├── train.py                   # 2-Phase training engine with callbacks
│   ├── predict.py                 # Multi-class inference engine & uncertainty handler
│   ├── gradcam.py                 # Explainable AI: Grad-CAM heatmap & overlay generator
│   ├── evaluation.py              # Test set evaluation, metrics, and confusion matrix
│   ├── flower_info.py             # 102 flower care & botanical profiles
│   └── __init__.py
│
├── assets/
│   └── sample_flowers/            # Built-in showcase flowers for 1-click demos
│
├── outputs/
│   ├── graphs/
│   │   └── training_history.png   # Measured training & validation loss/accuracy curves
│   ├── confusion_matrix.png       # 102-class high-resolution confusion matrix
│   └── reports/
│       ├── classification_report.txt # Full precision, recall, and F1 per class
│       ├── metrics.json           # Serialized test metrics
│       └── training_history.json  # Epoch-by-epoch loss & accuracy history
│
└── utils/
    ├── helpers.py                 # Image validation, PIL loaders, collection CRUD
    └── __init__.py
```

---

## 3. Deep Learning Methodology

### 3.1 MobileNetV2 Backbone
MobileNetV2 optimizes parameter efficiency through **Inverted Residual Blocks with Linear Bottlenecks**:
- **Depthwise Separable Convolutions:** Decomposes standard 3x3 convolutions into a spatial 3x3 depthwise filter and a 1x1 pointwise projection, reducing operations by approximately 8 to 9 times with minimal loss in representation.
- **Inverted Residuals:** Expands channel dimensionality by a factor of 6 in intermediate layers before spatial filtering, subsequently projecting back to low-dimensional linear bottlenecks.
- **Linear Bottleneck:** Omits nonlinear activations in the projection layer to prevent manifold destruction in low-dimensional spaces.

### 3.2 Two-Stage Training Pipeline
1. **Stage 1: Feature Extraction Head Warmup**
   - Base MobileNetV2 weights (ImageNet pretrained) remain frozen.
   - Classification head: `GlobalAveragePooling2D` -> `BatchNormalization` -> `Dropout(0.35)` -> `Dense(256, ReLU)` -> `Dropout(0.175)` -> `Dense(102, Softmax)`.
   - Trained with Adam optimizer (learning rate = 1e-3) using Sparse Categorical Cross-Entropy.
2. **Stage 2: Stratified Fine-Tuning**
   - Upper convolutional blocks (from layer 110 onwards) are unfrozen.
   - Batch normalization layers remain in inference mode to preserve moving statistics.
   - Trained with an attenuated learning rate (5e-5) to adapt specialized floral features while preventing catastrophic forgetting.

### 3.3 Callbacks Implemented
- **ModelCheckpoint:** Saves optimal model parameters based on validation accuracy (`models/best_model.keras`).
- **EarlyStopping:** Monitored validation loss to prevent overfitting and restored optimal weights.
- **ReduceLROnPlateau:** Dynamically reduced the learning rate by a factor of 0.5 when validation loss plateaued.

---

## 4. Explainable AI: Grad-CAM Formulation

Gradient-weighted Class Activation Mapping computes spatial importance gradients of the predicted class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The positive ReLU gate isolates features that positively increase the class score, filtering out non-flower background foliage.

---

## 5. Measured Experimental Results

Evaluated on the independent Oxford 102 held-out test split (1,229 unseen images):

| Metric | Score | Evaluation Condition |
| :--- | :--- | :--- |
| **Top-1 Test Accuracy** | **88.04%** | Evaluated on 1,229 unseen test images |
| **Top-3 Accuracy** | **96.82%** | Correct species within top 3 predictions |
| **Weighted Precision** | **89.89%** | Multi-class weighted precision |
| **Weighted Recall** | **88.04%** | Multi-class weighted recall |
| **Weighted F1-Score** | **0.8807** | Harmonized F1 across 102 flower categories |
| **Macro F1-Score** | **0.8763** | Unweighted average across all categories |
| **Final Validation Loss**| **0.4022** | Cross-entropy loss |
| **Model Size** | **27.7 MB** | Native Keras format (`models/best_model.keras`) |

---

## 6. Installation and Execution

### Prerequisites
- Python 3.10 or 3.11
- Operating System: Windows, macOS, or Linux

### Step 1: Clone Repository
```bash
git clone https://github.com/SayedaHaneen/PetalPop-AI.git
cd PetalPop-AI
```

### Step 2: Virtual Environment Setup
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 4: Run Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 7. Horticultural Care Database Coverage

The botanical database covers all 102 Oxford Flower species including:
*Pink Primrose, Lady's Slipper Orchid, Canterbury Bells, Sweet Pea, Tiger Lily, Moon Orchid, Bird of Paradise, Monkshood, King Protea, Purple Coneflower, Peruvian Lily, Anthurium, Hibiscus, Bougainvillea, Frangipani, Desert Rose, Camellia, Lotus, Water Lily*, and 83 additional species.

Each profile details:
- **Watering:** Frequency, soil drying thresholds, and seasonal dormancy allowances.
- **Sunlight:** Direct exposure versus filtered light and photoperiod requirements.
- **Soil Composition:** Substrate texture, pH balance, and drainage parameters.
- **Maintenance:** Pruning protocols, fertilizer schedules, and winterization.

---

## 8. Academic Submission Information

- **Project Title:** PetalPop AI: Intelligent Flower Recognition and Care Assistant
- **Framework:** TensorFlow 2.21+ / Keras 3.15+
- **Benchmark:** Oxford 102 Category Flower Dataset (Nilsback & Zisserman)
- **Detailed Research Report:** [UNIVERSITY_PROJECT_REPORT.md](UNIVERSITY_PROJECT_REPORT.md)
