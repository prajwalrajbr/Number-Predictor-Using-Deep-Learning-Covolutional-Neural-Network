""" Completely written with Visualization on Google Colab at 'https://colab.research.google.com/drive/1IgLdLzqwm5BcEPyEND0J-MP26imv9FDi?usp=sharing' """

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import numpy as np
from keras.utils import np_utils

scores = []


class numberPredictorCNN:
    def trainModel(self):
        global scores

        # Load the mnist dataset and store it
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

        # Flattening images into 28*28
        train_dataset = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
        test_dataset = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

        # Normalizing to Sigmoid function values between 0 and 1
        train_dataset = train_dataset / 255
        test_dataset = test_dataset / 255

        # Encoding the classes to categorical data
        train_labels = np_utils.to_categorical(y_train)
        test_labels = np_utils.to_categorical(y_test)
        no_of_classes = test_labels.shape[1]

        # Check if we have previously saved our model
        try:
            model = tf.keras.models.load_model('number_predictor.h5')
            scores = model.evaluate(test_dataset, test_labels, verbose=0)
            del model
        # Create a CNN model and Save it as a HDF5 file 'number_predictor.h5'
        except:
            # Creating the Convolutional Neural Network Layers
            def build_model():
                model = Sequential()
                model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
                model.add(MaxPooling2D())
                model.add(Conv2D(15, (3, 3), activation='relu'))
                model.add(MaxPooling2D())
                model.add(Dropout(0.2))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(no_of_classes, activation='softmax'))
                # Compile model
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                return model

            # Building the Convolutional Neural Network model
            model = build_model()

            # Training the CNN model
            model.fit(train_dataset, train_labels, validation_data=(test_dataset, test_labels), epochs=10,
                      batch_size=200)

            scores = model.evaluate(test_dataset, test_labels, verbose=0)
            # Creates a HDF5 file 'number_predictor.h5'
            model.save('number_predictor.h5')
            del model

    def predictNumber(self, test):
        # Predicting Our Drawing
        model = tf.keras.models.load_model('number_predictor.h5')
        test = np.asarray(test, dtype=float)
        predict = model.predict(test.reshape(1, 28, 28, 1))
        return predict.argmax()


if __name__ == "__main__":
    CNN_model = numberPredictorCNN()
    CNN_model.trainModel()

    print("Test Loss", scores[0])
    print("Test Accuracy", scores[1])

else:
    print("numberPredictorCNN is successfully imported")

""" Completely written with Visualization on Google Colab at 'https://colab.research.google.com/drive/1IgLdLzqwm5BcEPyEND0J-MP26imv9FDi?usp=sharing' """
