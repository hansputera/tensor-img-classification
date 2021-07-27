import pathlib
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
from sys import exit
import datetime as dt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

sources = "flower_photos"
if os.path.exists(sources) == False:
    print("Execute dl_trainset.py first")
    exit(0)

data_dir = pathlib.Path(sources)

# Image count
img_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count: " + str(img_count))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names

def train():
    global train_ds, val_ds
    for image_batch, label_batch in train_ds.take(1):
        print('Image batch shape: ', image_batch.shape)
        print('Label batch shape: ', label_batch.shape)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Standarize data
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    num_classes = 5
    # Creating model
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    model.summary()

    epochs = 15
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Visualizing train results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_ranges = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_ranges, acc, label='Training Accuracy')
    plt.plot(epochs_ranges, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_ranges, loss, label='Training Loss')
    plt.plot(epochs_ranges, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Getting current time
    now = dt.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    plt.savefig(f"visual/{now}.png")

    # Saving model
    print("Saving model")
    model.save("models/flowers_model.h5")