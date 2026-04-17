# Brain Tumor Classification App

This project is a deep learning application that predicts the presence of a brain tumor from MRI images using a ResNet model.

## Project Description

The objective of this project is to build a robust binary classification model capable of distinguishing between:
- Tumor
- No Tumor

The model was trained on MRI images with preprocessing techniques and deployed using Streamlit.

## Model

- Architecture: ResNet
- Type: Binary classification
- Input size: 224x224
- Preprocessing:
  - Image resizing
  - Normalization (0–1)
  - (Optional: CLAHE if applied)
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

## How to run the application

### 1. Install dependencies

```bash
pip install -r requirements.txt

## Authors

- Carole Mbomegni (GitHub: @tonusername)
- Nom 2 (GitHub: @username2)
- Nom 3
- Nom 4
- Nom 5
- Nom 6