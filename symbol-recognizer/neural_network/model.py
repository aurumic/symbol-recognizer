import tensorflow as tf
import numpy as np
import scipy.io

dataset_path = './datasets/emnist-balanced.mat'

data = scipy.io.loadmat(dataset_path)

x_train = data['dataset']['train'][0][0]['images'][0][0]
y_train = data['dataset']['train'][0][0]['labels'][0][0]
x_test = data['dataset']['test'][0][0]['images'][0][0]
y_test = data['dataset']['test'][0][0]['labels'][0][0]

x_train = x_train.reshape((-1, 28, 28, 1), order='A')
x_test = x_test.reshape((-1, 28, 28, 1), order='A')

# X_train = x_train.astype(np.float32) / 255.0
# X_test = x_test.astype(np.float32) / 255.0

y_train = y_train.astype(np.int32).flatten()
y_test = y_test.astype(np.int32).flatten()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(buffer_size=1024)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

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
        tf.keras.layers.Dense(47, activation='softmax'),
    ]
)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(train_dataset, epochs=10, validation_data=test_dataset)

model.save("model.keras")
