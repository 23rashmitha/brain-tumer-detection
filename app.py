import os
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Disable GPU to avoid CUDA errors in Hugging Face Spaces
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the trained model
model = load_model("brain_tumor_model.h5")

# Class labels (must match your training order)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 🔍 Find last Conv2D layer
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

# 🔥 Grad-CAM implementation
def get_gradcam(img_array, model, layer_name):
    # Use model.input to ensure correct input structure
    grad_model = tf.keras.models.Model([model.input], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        # Wrap input in a list to match expected structure
        conv_outputs, predictions = grad_model([img_array.astype(np.float32)])
        # Handle case where predictions is a list of tensors
        if isinstance(predictions, list) and len(predictions) > 0:
            predictions = predictions[0]
        pred_index = tf.argmax(predictions, axis=-1)[0]  # Get index for the first batch
        class_channel = tf.gather(predictions, pred_index, axis=-1)  # Safe tensor indexing

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# 🧠 Gradio prediction function
def predict(image_input):
    if image_input is None:
        return "No image uploaded", None

    # Convert RGBA to RGB if needed
    if image_input.shape[2] == 4:
        image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)

    # Preprocess image
    img = cv2.resize(image_input, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0).astype(np.float32)  # Ensure float32

    # Get prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100

    # Generate Grad-CAM
    last_conv = get_last_conv_layer_name(model)
    heatmap = get_gradcam(img_array, model, last_conv)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", overlay

# 🚀 Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Brain MRI (JPG or PNG)"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title="Brain Tumor Classifier with Grad-CAM",
    description="Upload a brain MRI to detect tumor type (glioma, meningioma, notumor, pituitary) and visualize Grad-CAM heatmap."
)

if __name__ == "__main__":
    interface.launch()
