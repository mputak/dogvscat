import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

batch_size = 32
img_height = 200
img_width = 200
epochs = 40
AUTOTUNE = tf.data.AUTOTUNE

print(tf.config.list_physical_devices('GPU'))

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="archive",
    validation_split=0.2,
    seed=42,
    image_size=(img_height, img_width),
    subset="both",
)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=4,verbose=1),
    tf.keras.callbacks.ModelCheckpoint("best_model_augmented.h5", save_best_only=True, verbose=1)]

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
