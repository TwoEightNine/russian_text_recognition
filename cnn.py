import os
import pickle

from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer

MODEL = 'model.hdf5'
CLASSES = 'classes.data'


class CNN(object):

    def __init__(self, image_size=32, letters=43):
        self.image_size = image_size
        self.letters_count = letters
        if os.path.isfile(MODEL):
            self.model = load_model(MODEL)
        if os.path.isfile(CLASSES):
            self.classes = pickle.load(open(CLASSES, 'rb'))

    def __create_model(self):
        network = Sequential()

        network.add(Conv2D(
            filters=32,
            kernel_size=3,
            input_shape=(self.image_size, self.image_size, 1),
            padding='same',
            activation='relu'
        ))
        network.add(Conv2D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu'
        ))
        network.add(Conv2D(
            filters=32,
            kernel_size=5,
            padding='same',
            strides=2,
            activation='relu'
        ))
        network.add(Dropout(0.4))

        network.add(Conv2D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation='relu'
        ))
        network.add(Conv2D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation='relu'
        ))
        network.add(Conv2D(
            filters=64,
            kernel_size=5,
            padding='same',
            strides=2,
            activation='relu'
        ))
        network.add(Dropout(0.4))

        network.add(Flatten())
        network.add(Dense(units=128, activation='relu'))
        network.add(Dropout(0.4))

        network.add(Dense(units=self.letters_count, activation='softmax'))

        network.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        return network

    def train_and_save(self, image_file_names, letters):
        lb = LabelBinarizer()
        letters = lb.fit_transform(letters)
        self.classes = list(lb.classes_)
        pickle.dump(self.classes, open(CLASSES, 'wb'))
        # letters = np_utils.to_categorical(letters)
        images = np.array(
            [np.array(Image.open(name)).reshape((self.image_size, self.image_size)) for name in image_file_names])
        images = images.reshape((-1, self.image_size, self.image_size, 1))
        images = images / 255
        print(images)

        imagegen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=1,
            height_shift_range=1,
            zoom_range=0.1
        )
        network = self.__create_model()
        network.fit_generator(
            imagegen.flow(images, letters, batch_size=32),
            epochs=100,
            verbose=1,
            steps_per_epoch=images.shape[0] // 32,
            # callbacks=[EarlyStopping(monitor='val_loss', patience=0, restore_best_weights=True)],
            validation_data=(images, letters)
        )
        network.save(MODEL)
        self.model = network

    def predict(self, image):
        image = np.array(image).reshape((-1, self.image_size, self.image_size, 1)) / 255
        prob = self.model.predict(image)
        pos = np.argmax(prob)
        return self.classes[pos], prob[0][pos]
