import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.models import model_from_json
import numpy

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def show_image(image, prediction):
    plt.figure()
    plt.imshow(image)
    plt.title(CLASS_NAMES[prediction])
    plt.axis("off")
    plt.show()

if __name__ == '__main__':

    CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    BATCH_SIZE = 1
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_dir = r"C:\Users\Nathan Jiang\Desktop\Machine Learning\Flowers\test_images"
    IMAGE_COUNT = len(os.listdir(data_dir))

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    list_ds = tf.data.Dataset.list_files('{}/*'.format(data_dir))
    processed_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    processed_ds = processed_ds.batch(BATCH_SIZE)

    for i, single_ds in processed_ds.enumerate():
        prediction = np.argmax(loaded_model.predict(single_ds)[0])
        show_image(next(iter(single_ds)), prediction)
        if i == IMAGE_COUNT - 1:
            break
