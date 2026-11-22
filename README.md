# Crop Disease Detection

A machine-learning web app that identifies crop diseases from leaf images using a trained TensorFlow/Keras model and Streamlit.

## Features

- Upload a leaf image
- Get the predicted disease class with confidence
- Supports the crop/disease classes defined in `class_indices.json`

## Tech Stack

Python · TensorFlow/Keras · Streamlit · NumPy · Pillow

## Files

- `main.py` — Streamlit app and prediction pipeline
- `CropDiseaseDetection.ipynb` — model training notebook
- `class_indices.json` — class label mapping
- `requirements.txt` — dependencies

## Getting Started

```bash
pip install -r requirements.txt
streamlit run main.py
```

> The trained model file (`trained_model/plant_disease_model.h5`) must be present before running the app.
