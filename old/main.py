import base64
import binascii
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.datasets import mnist

app = Flask(__name__)
CORS(app)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(62, activation='softmax'),
    ]
)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)


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

    prediction = int(model.predict(tf_image).argmax())

    return jsonify({"prediction": prediction}), 200


if __name__ == "__main__":
    app.run(debug=True)
