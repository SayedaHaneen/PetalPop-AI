# PetalPop AI 🌸: Intelligent Flower Recognition and Care Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-4B8BBE?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Oxford%20102%20Flowers-green?style=for-the-badge)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

> **A University-Level Deep Learning & Explainable Computer Vision Project**  
> *Identifies 102 flower species with MobileNetV2 Transfer Learning, visualizes decision attention using Grad-CAM, provides tailored botanical care guidelines, and saves your personalized garden collection.*

---

## 🌺 1. Project Overview

**PetalPop AI** is an end-to-end intelligent botanical recognition system and plant care companion. Fine-grained visual categorization of flowers is notoriously difficult due to extreme intra-class variability (differences in color stages, petal count, perspective, lighting) and subtle inter-class differences between related botanical species.

PetalPop AI solves this through **Transfer Learning & Fine-Tuning with MobileNetV2**, achieving high classification accuracy while maintaining lightweight, real-time efficiency. To address the black-box nature of deep neural networks, PetalPop AI incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually highlight the anatomical regions (petals, stamen, corolla) influencing each prediction.

### Key Capabilities:
1. 📸 **Instant Image Identification:** Drag & drop any flower photograph (JPG, JPEG, PNG).
2. 🎯 **Confidence Scoring & Top-3 Candidates:** Real-time softmax probability distributions.
3. ⚠️ **Uncertainty Alert:** Warns users if image quality or ambiguity causes low model confidence.
4. 🧠 **Explainable AI (Grad-CAM):** Visualizes the exact convolutional feature maps guiding the AI.
5. 💧 **Botanical Care Assistant:** Authoritative watering, sunlight, soil, and care instructions for all 102 classes.
6. 🌺 **Personal Garden Collection:** Locally saves identified specimens with personal field notes.
7. 📊 **Academic Evaluation Dashboard:** Displays real training curves, confusion matrix, and precision/recall/F1 metrics.

---

## 🏗️ 2. System Architecture

```
PetalPop_AI/
│
├── app.py                         # Interactive Streamlit Web Interface
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive academic & setup documentation
├── UNIVERSITY_PROJECT_REPORT.md   # Full university research project report
│
├── models/
│   └── best_model.keras           # Trained MobileNetV2 model weights
│
├── data/
│   ├── raw/                       # Original Oxford 102 tgz archive & labels
│   ├── processed/                 # Stratified train/, val/, test/ folders
│   └── collection.json            # Persistent user garden collection
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Download, extraction, augmentation pipelines
│   ├── model.py                   # MobileNetV2 architecture & head design
│   ├── train.py                   # 2-Phase training & fine-tuning engine
│   ├── predict.py                 # Inference engine, top-k, uncertainty handler
│   ├── gradcam.py                 # Grad-CAM heatmap & overlay generator
│   ├── evaluation.py              # Metrics, confusion matrix, classification report
│   └── flower_info.py             # 102-species botanical care database
│
├── outputs/
│   ├── graphs/
│   │   └── training_history.png   # Accuracy & loss curves over epochs
│   ├── confusion_matrix.png       # High-resolution 102-class confusion matrix
│   └── reports/
│       ├── classification_report.txt # Per-class precision, recall, and F1
│       └── metrics.json           # Serialized summary metrics
│
└── utils/
    └── helpers.py                 # Validation, PIL helpers, collection JSON CRUD
```

---

## 🔬 3. Deep Learning Methodology

### 3.1 MobileNetV2 Backbone
MobileNetV2 uses **Inverted Residual Blocks with Linear Bottlenecks**:
- **Depthwise Separable Convolutions:** Decomposes standard $3 \times 3$ convolutions into a spatial $3 \times 3$ depthwise filter and a $1 \times 1$ pointwise projection, reducing FLOPs by ~8-9× with minimal accuracy loss.
- **Inverted Residuals:** Expands channel capacity by factor $t=6$ in intermediate layers before depthwise filtering, then projects back to low-dimensional linear bottlenecks.
- **Linear Bottleneck:** Replaces nonlinear activation in the projection layer with a linear activation to prevent information loss on low-dimensional manifolds.

### 3.2 Two-Stage Training Pipeline
1. **Stage 1 (Warmup / Feature Extraction):**
   - Base MobileNetV2 weights (ImageNet pretrained) are frozen.
   - Classification head: `GlobalAveragePooling2D` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.35)` $\rightarrow$ `Dense(256, activation='relu')` $\rightarrow$ `Dropout(0.175)` $\rightarrow$ `Dense(102, activation='softmax')`.
   - Trained with Adam optimizer ($\alpha = 10^{-3}$) and Sparse Categorical Cross-Entropy.
2. **Stage 2 (Stratified Fine-Tuning):**
   - Upper convolutional blocks (from layer 100 onwards) are unfrozen.
   - Batch normalization layers remain frozen in inference mode for stability.
   - Trained with a reduced learning rate ($\alpha = 5 \times 10^{-5}$) to adapt higher-level semantic features without catastrophic forgetting.

### 3.3 Callbacks Employed
- **`ModelCheckpoint`:** Saves the optimal model state based on validation accuracy (`best_model.keras`).
- **`EarlyStopping`:** Prevents overfitting by monitoring validation loss with a patience threshold and restores best weights.
- **`ReduceLROnPlateau`:** Dynamically scales down learning rate by factor 0.5 when validation loss plateaus.

---

## 🧠 4. Explainable AI: Grad-CAM

Gradient-weighted Class Activation Mapping computes the spatial gradients of the winning class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

Applying a positive ReLU filter ensures that the generated heatmap only highlights floral regions that increase the class score, filtering out non-flower background foliage.

---

## 📊 5. Experimental Results

Evaluated on the independent Oxford 102 held-out test split (1,229 unseen images):

| Metric | Score |
| :--- | :--- |
| **Top-1 Test Accuracy** | **89.4%** |
| **Top-3 Test Accuracy** | **96.8%** |
| **Weighted Precision** | **90.3%** |
| **Weighted Recall** | **89.4%** |
| **Weighted F1-Score** | **0.896** |
| **Macro F1-Score** | **0.881** |

---

## 🚀 6. Installation and Execution

### Prerequisites
- Python 3.10 or 3.11
- Windows, macOS, or Linux

### Step 1: Clone Repository
```bash
git clone https://github.com/SayedaHaneen/PetalPop-AI.git
cd PetalPop-AI
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 5: (Optional) Run Dataset Preparation & Training
```bash
# Download and organize Oxford 102 dataset
python download_and_prep.py

# Train MobileNetV2 model
python src/train.py --epochs_head 5 --epochs_finetune 8 --batch_size 32
```

---

## 🌿 7. Flower Care Database Coverage

The database covers all 102 Oxford Flower species including:
- *Pink Primrose, Lady's Slipper Orchid, Canterbury Bells, Sweet Pea, Tiger Lily, Moon Orchid, Bird of Paradise, Monkshood, King Protea, Purple Coneflower, Peruvian Lily, Anthurium, Hibiscus, Bougainvillea, Frangipani, Desert Rose, Camellia, Lotus, Water Lily*, and 83 more!

Each entry provides:
- 💧 **Watering frequency & seasonal adjustments**
- ☀️ **Sunlight exposure & photoperiod**
- 🌱 **Soil composition, pH, and drainage**
- ✂️ **Pruning, fertilization, and winterizing**

---

## 📜 8. Academic Submission Details

- **Project:** PetalPop AI: Intelligent Flower Recognition and Care Assistant
- **Framework:** TensorFlow 2.15+ / Keras 3.0+
- **Evaluation Benchmark:** Oxford 102 Flowers (Nilsback & Zisserman)
- **Report Document:** [UNIVERSITY_PROJECT_REPORT.md](UNIVERSITY_PROJECT_REPORT.md)

---
*Developed with love and deep learning 🌸*
