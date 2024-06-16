import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)
    return image, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

dataset_path = "./datasets/emnist-byclass.mat"

data = scipy.io.loadmat(dataset_path)

x_train = data["dataset"]["train"][0][0]["images"][0][0]
y_train = data["dataset"]["train"][0][0]["labels"][0][0]
x_test = data["dataset"]["test"][0][0]["images"][0][0]
y_test = data["dataset"]["test"][0][0]["labels"][0][0]

x_train = x_train.reshape((-1, 28, 28, 1), order="A")
x_test = x_test.reshape((-1, 28, 28, 1), order="A")

y_train = y_train.astype(np.int32).flatten()
y_test = y_test.astype(np.int32).flatten()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = (
    train_dataset.map(preprocess_image)
    .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=10000)
    .batch(128)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
test_dataset = test_dataset.map(preprocess_image).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(62, activation="softmax", dtype='float32')
])

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

model.fit(
    train_dataset, 
    epochs=50, 
    validation_data=test_dataset, 
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

model.save("model.keras")
