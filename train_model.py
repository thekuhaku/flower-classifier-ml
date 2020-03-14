import tensorflow as tf
import pathlib
import glob
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.models import model_from_json

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

if __name__ == '__main__':

    SHUFFLE_BUFFER = 32
    DEV_SIZE = 100
    EPOCHS = 50 
    BATCH_SIZE = 16
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)
    # print("Path to file:", data_dir)
    # image_count = len(list(data_dir.glob('*/*.jpg')))
    # print("Image count", image_count)
    
    CLASS_NAMES = list([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    # print("Class names:", CLASS_NAMES)

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    labeled_ds = labeled_ds.shuffle(SHUFFLE_BUFFER)
    dev_ds = labeled_ds.take(DEV_SIZE)
    dev_ds = dev_ds.batch(BATCH_SIZE)
    train_ds = labeled_ds.skip(DEV_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, epochs=EPOCHS)

    # save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
