import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Define the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(working_dir, 'trained_model', 'plant_disease_model.h5')
model = tf.keras.models.load_model(model_path)

# Get the input shape of the model
input_shape = model.input_shape
print(f"Model input shape: {input_shape}")  # Debugging

# Load class indices
class_indices_path = os.path.join(working_dir, 'class_indices.json')
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
print("Class indices:", class_indices)  # Debugging

# Function to preprocess image
def load_and_preprocess_image(image, target_size=(150, 150)):
    """
    Preprocesses the image to match the model's input requirements.
    Args:
        image: Uploaded file (file-like object) or image path.
        target_size: Target size of the image (height, width).
    Returns:
        Preprocessed image as a numpy array with shape (1, 150, 150, 3).
    """
    if isinstance(image, str):  # If image is a file path
        img = Image.open(image)
    else:  # If image is a file-like object (uploaded file)
        img = Image.open(image)
    
    # Resize and convert to RGB (in case of grayscale images)
    img = img.resize(target_size).convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0  # You might need to adjust this if your model was trained differently
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict
def predict_image_class(model, image, class_indices):
    """
    Predicts the class of the input image.
    Args:
        model: Loaded Keras model.
        image: Uploaded file (file-like object) or image path.
        class_indices: Dictionary mapping class indices to class names.
    Returns:
        Predicted class name.
    """
    preprocessed_img = load_and_preprocess_image(image)
    
    # Debugging: Print the shape and values of the preprocessed image
    print("Preprocessed image shape:", preprocessed_img.shape)
    print("Preprocessed image values (min, max):", preprocessed_img.min(), preprocessed_img.max())
    
    # Make the prediction
    predictions = model.predict(preprocessed_img)
    
    # Debugging: Print the raw predictions (probabilities for each class)
    print("Raw predictions (probabilities):", predictions)
    
    # Check if there's an overfitting issue (e.g., always predicting one class)
    top_n_predictions = np.argsort(predictions[0])[::-1][:3]  # Top 3 predictions
    print(f"Top 3 predictions (indices): {top_n_predictions}")
    print(f"Top 3 predictions (class names): {[class_indices[str(i)] for i in top_n_predictions]}")
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Debugging: Verify the predicted class index and the corresponding class name
    print(f"Predicted class index: {predicted_class_index}")
    
    # Map the predicted index to the class name
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name

# Streamlit App
st.title('Crop Disease Detection')

# Upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    predicted_class = predict_image_class(model, uploaded_image, class_indices)
    
    # Display prediction result
    st.success(f"Prediction: {predicted_class}")
