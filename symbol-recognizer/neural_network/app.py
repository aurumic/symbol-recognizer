import base64
import binascii
import string
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model.keras")

@app.route("/upload", methods=["POST"])
def process_image():
    print("post get")

    data = request.get_json()
    image_data = data["imageURL"]
    image_data = image_data.split(",")[1]
    image_data += "=" * ((4 - len(image_data) % 4) % 4)

    try:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
    except binascii.Error as e:
        return jsonify({"error": str(e)}), 400

    image = image.convert("L")
    image = image.resize((28, 28))

    tf_image = np.array(image).reshape((1, 28, 28, 1))
    
    label_map = string.digits + string.ascii_uppercase + string.ascii_lowercase
    label_map = ''.join(sorted(set(label_map)))

    predictions = model.predict(tf_image)
    top_5_indices = np.argsort(predictions[0])[::-1][:5]
    top_5_characters = [label_map[index] for index in top_5_indices]
    top_5_probabilities = predictions[0][top_5_indices]

    print(f"predictions: {predictions}")

    top_5_predictions = [{"character": top_5_characters[i], "probability": float(top_5_probabilities[i])} for i in range(5)]

    return jsonify(top_5_predictions), 200

if __name__ == "__main__":
    app.run(debug=True)
