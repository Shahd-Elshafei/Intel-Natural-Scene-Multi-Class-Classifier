# Intel Natural Scene Multi-Class Classifier

An end-to-end Computer Vision project leveraging transfer learning with MobileNetV2 to classify natural landscapes and urban environments into 6 distinct categories.

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Model Performance Summary](#model-performance-summary)
- [Installation & Setup](#installation--setup)
- [Usage Instructions](#usage-instructions)
  - [1. Training & Evaluation](#1-training--evaluation)
  - [2. Loading Saved Model for Inference](#2-loading-saved-model-for-inference)
  - [3. Grad-CAM Visual Explainability](#3-grad-cam-visual-explainability)
- [Bonus Implementations](#bonus-implementations)

## Overview

This project builds and evaluates deep learning models to perform automated natural scene recognition.

- **Target Classes (6):** buildings, forest, glacier, mountain, sea, street
- **Input Resolution:** 150×150×3 RGB images
- **Best Model:** MobileNetV2 fine-tuned with inline data augmentation and dynamic learning rate scheduling.

## Dataset Structure

The dataset expected by the pipeline should follow this directory structure:

```
dataset/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── ...
└── seg_pred/
    └── seg_pred/      # Unlabelled external images for prediction
```

## Model Performance Summary

| Architecture | Input Size | Parameters | Test Accuracy | Macro F1-Score |
|---|---|---|---|---|
| Baseline CNN | 150×150 | 5,698,630 | 83.50% | 0.8346 |
| MobileNetV2 (Transfer) | 150×150 | 1,534,086 | 90.30% | 0.9047 |

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Recommended environment: Jupyter Notebook or Google Colab (GPU-accelerated)

### Install Dependencies

Clone or download this repository, then install the required dependencies:

```bash
pip install tensorflow numpy pandas matplotlib pillow opencv-python
```

## Usage Instructions

### 1. Training & Evaluation

Open `notebook.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially to:

- Execute Exploratory Data Analysis (EDA).
- Build data pipelines with inline augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, etc.).
- Train the baseline CNN and MobileNetV2 models.
- Generate classification reports and confusion matrices.

### 2. Loading Saved Model for Inference

The trained MobileNetV2 model is exported as `intel_scene_classifier_mobilenetv2.keras`. You can load and use it directly without retraining:

```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load saved model and define class ordering
reloaded_model = tf.keras.models.load_model("intel_scene_classifier_mobilenetv2.keras")
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_external_image(image_path, model, class_names):
    # Preprocess image
    raw_img = Image.open(image_path).convert('RGB')
    resized_img = raw_img.resize((150, 150))
    img_array = np.expand_dims(np.array(resized_img), axis=0)

    # Predict class
    predictions = model.predict(img_array, verbose=0)
    top_idx = np.argmax(predictions[0])
    predicted_class = class_names[top_idx]
    confidence = predictions[0][top_idx] * 100

    # Display result
    plt.figure(figsize=(4, 4))
    plt.imshow(raw_img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

# Example usage
predict_external_image("path_to_your_image.jpg", reloaded_model, class_names)
```

### 3. Grad-CAM Visual Explainability

To visualize which image features the model uses to make predictions:

```python
# Run the generate_gradcam() function provided in the notebook
heatmap = generate_gradcam(img_array, reloaded_model)

# Overlay heatmap on raw image
heatmap_resized = cv2.resize(heatmap, (150, 150))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(np.array(raw_img), 0.6, heatmap_colored, 0.4, 0)
```

## Bonus Implementations

- **Model Saving:** Model exported to a single portable `.keras` binary containing architecture, weights, and compilation state.
- **Grad-CAM Explainability:** Integrated Gradient-weighted Class Activation Mapping to generate heatmaps highlighting decision focus regions.
