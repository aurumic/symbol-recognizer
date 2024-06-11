import base64
from io import BytesIO
from random import randint

import matplotlib.pyplot as plt
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
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)


@app.route("/upload", methods=["POST"])
def upload_image():
    print("post get")

    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert("L")
    image = image.resize((28, 28))

    tf_image = np.array(image).reshape((1, 28, 28, 1))

    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.title(f"predicted: {model.predict(tf_image).argmax()}")
    plt.show()

    return jsonify({"message": "Image received"}), 200


app.run()
