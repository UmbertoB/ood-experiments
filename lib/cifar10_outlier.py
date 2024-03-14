import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def get():
    dataset_name = "cifar10_corrupted"
    ds = tfds.load(dataset_name, split="test", shuffle_files=False, as_supervised=True)

    x_data = []
    y_data = []

    for image, label in ds:
        image = tf.image.resize(image, (32, 32))
        image = image.numpy()
        label = label.numpy()
        x_data.append(image)
        y_data.append(label)

    # Convert lists to numpy arrays
    x_data = np.array(x_data)
    x_data = x_data.astype('float32')
    x_data = x_data / 255

    y_data = np.array(y_data)

    return x_data, y_data
