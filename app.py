from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load the MNIST model
MODEL_PATH = "cnn_mnist_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')  # Serves the frontend

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.LANCZOS)
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    # Predict the digit
    predictions = model.predict(image)
    predicted_digit = int(np.argmax(predictions))
    probabilities = predictions[0].tolist()

    return jsonify({"digit": predicted_digit, "probabilities": probabilities})

@app.route('/save', methods=['POST'])
def save_image():
    # Get the image from the request
    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode and save the image
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("L")
    save_path = os.path.join("saved_images", "digit.png")
    os.makedirs("saved_images", exist_ok=True)
    image.save(save_path)

    return jsonify({"message": f"Image saved at {save_path}"}), 200

if __name__ == '__main__':
    app.run(debug=True)
