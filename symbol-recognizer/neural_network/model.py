import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import scipy.io

def preprocess_image(image, label):
    return image, label

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

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.map(preprocess_image).shuffle(buffer_size=10000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = models.Sequential()

# Input Layer
model.add(layers.Input(shape=(28, 28, 1)))

# Convolutional Block 1
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
# Convolutional Block 2
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
# Convolutional Block 3
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
# Flatten and Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(47, activation='softmax'))



model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(train_dataset, epochs=10, batch_size=128, validation_data=test_dataset)

model.save("model.keras")
