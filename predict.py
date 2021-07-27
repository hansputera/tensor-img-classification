import tensorflow as tf
import numpy as np
from main import class_names, img_height, img_width
from os import path
from sys import exit
from shutil import move

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
print("Moving predict set")
move(sunflower_path, "predict_test.jpg")

model_path = "models/flowers_model.h5"
if path.exists(model_path) == False:
    print("Train model first")
    exit(0)

print("Load trained model")
model = tf.keras.models.load_model(model_path)

img = tf.keras.preprocessing.image.load_img("predict_test.jpg", target_size=(img_width, img_height))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Predicting
predictions = model.predict(img_array)

for prediction in predictions:
    score = tf.nn.softmax(prediction)
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    )