import pathlib
import tensorflow as tf
from shutil import move

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

print("Moving datasets")
move(data_dir, pathlib.Path("flower_photos").resolve())