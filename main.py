from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)

# Load the model and class names
MODEL = tf.keras.models.load_model("./initial_filter_version2")
CLASS_NAMES = ['not a road', 'road']

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.route("/ping", methods=['GET'])
def ping():
    return "Hello, I'm alive"


@app.route("/predict", methods=['POST'])
def predict():
    # Check if the request contains any data
    if request.data:
        # Assuming the data is in byte stream format
        byte_data = request.data

        # Read the byte data as an image using PIL
        image = Image.open(BytesIO(byte_data))

        # Display the image (optional)
        # image.show()
        image = image.resize((256, 256))  # Resize the image
        image = np.array(image)
        img_batch = np.expand_dims(image, 0)
        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        print('predicted_class', predicted_class)
        print('confidence', float(confidence * 100))
        # return jsonify({
        #     'predicted_class': predicted_class,
        #     'confidence': float(confidence * 100)
        # })
        if predicted_class ==

    else:
        # If no data is sent in the request, return an error response
        return "No data received", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
