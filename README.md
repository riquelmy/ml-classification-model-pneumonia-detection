# 🫁 Pneumonia Detection using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images as **Pneumonia** or **Normal** with ~91% accuracy. Includes an interactive **Streamlit dashboard** for model exploration and live predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Streamlit Dashboard](#streamlit-dashboard)
- [References](#references)

## Overview

Pneumonia is an inflammatory condition of the lungs that affects millions worldwide. Early and accurate diagnosis through chest X-rays can significantly improve patient outcomes. This project automates pneumonia detection using a CNN trained on pediatric chest X-ray images.

**Key Features:**
- 5-layer CNN with BatchNormalization and Dropout
- Data augmentation to handle class imbalance
- Interactive Streamlit dashboard for presentations and demos
- Upload any chest X-ray for real-time prediction

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Pneumonia | Normal | Total |
|-------|-----------|--------|-------|
| Train | 3,875     | 1,341  | 5,216 |
| Test  | 390       | 234    | 624   |
| Val   | 8         | 8      | 16    |

- **Image type:** Anterior-posterior chest X-rays (JPEG, grayscale)
- **Patient demographics:** Pediatric patients aged 1–5 years
- **Source institution:** Guangzhou Women and Children's Medical Center
- **Quality control:** Images screened for quality; diagnoses graded by two expert physicians and verified by a third

## Model Architecture

Sequential CNN with 5 convolutional blocks:

```
Input (150×150×1)
    │
    ├── Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2)
    ├── Conv2D(64, 3×3) → Dropout(0.1) → BatchNorm → MaxPool(2×2)
    ├── Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2)
    ├── Conv2D(128, 3×3) → Dropout(0.2) → BatchNorm → MaxPool(2×2)
    ├── Conv2D(256, 3×3) → Dropout(0.2) → BatchNorm → MaxPool(2×2)
    │
    ├── Flatten → Dense(128, ReLU) → Dropout(0.2)
    └── Dense(1, Sigmoid) → Output
```

| Parameter | Value |
|-----------|-------|
| Total parameters | 1,246,401 |
| Trainable parameters | 1,245,313 |
| Optimizer | RMSprop |
| Loss function | Binary Crossentropy |
| Learning rate scheduler | ReduceLROnPlateau (patience=2, factor=0.3) |
| Epochs | 12 |
| Batch size | 32 |

**Data Augmentation:**
- Rotation: ±30°
- Zoom: 20%
- Width/Height shift: 10%
- Horizontal flip: enabled

## Results

### Test Set Performance (624 images)

| Metric | Value |
|--------|-------|
| **Accuracy** | **91.3%** |
| **Loss** | 0.2524 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Pneumonia | 93% | 93% | 93% | 390 |
| Normal | 88% | 88% | 88% | 234 |
| **Weighted Avg** | **91%** | **91%** | **91%** | **624** |

### Confusion Matrix

|  | Predicted Pneumonia | Predicted Normal |
|--|---------------------|------------------|
| **Actual Pneumonia** | 363 | 27 |
| **Actual Normal** | 27 | 207 |

## Project Structure

```
modelagem_comp_2/
├── app.py                                          # Streamlit dashboard application
├── pneumonia-detection-using-cnn-92-6-accuracy.ipynb  # Training notebook
├── pneumonia_cnn_model.keras                       # Saved trained model
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone or download the project:**
   ```bash
   cd modelagem_comp_2
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/macOS
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model (Jupyter Notebook)

Open and run `pneumonia-detection-using-cnn-92-6-accuracy.ipynb` in Jupyter or VS Code. The notebook will:
1. Download the dataset automatically via `kagglehub`
2. Preprocess and augment the images
3. Train the CNN for 12 epochs
4. Evaluate on the test set
5. Save the model to `pneumonia_cnn_model.keras`

### Running Predictions (Streamlit)

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Streamlit Dashboard

The interactive dashboard includes 7 pages:

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Project summary, pneumonia background, key metrics |
| 📊 **Dataset Explorer** | Dataset splits, class distribution charts, sample image browser |
| 🧠 **Model Architecture** | Layer-by-layer breakdown, parameter counts, augmentation strategy |
| 📈 **Training Results** | Interactive accuracy/loss curves, learning rate schedule |
| 🔬 **Test Evaluation** | Classification report, confusion matrix, correct/incorrect examples |
| 🩻 **Predict Your X-Ray** | Upload any X-ray for live prediction with confidence gauge |
| 🖼️ **Sample Predictions** | Random validation samples with prediction results |

## References

- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- **Paper:** Kermany, D.S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122-1131.

---

> ⚕️ **Disclaimer:** This project is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis.
