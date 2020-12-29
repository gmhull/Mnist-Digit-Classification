import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

# This function is used to train a model with the goal of classifying 28x28 pixel images to a specific digit.
def train():
    # Load the dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the values from the
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential() # This creates the neural network.  Sequential because its feed forward.

    model.add(tf.keras.layers.Flatten()) # This turns the 28x28 picture into a 784x1

    # Dense layers are used to make fully connected layers.
    # The activation function is Rectified Linear.  {0 if x < 0: x if x > 0}
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # Softmax activation function shows the probability of selecting the item.
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # This is to compile the results
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Run the model for the given number of epochs
    model.fit(x_train, y_train, epochs=3)

    # Save the model for future use
    model.save('num_classification.model')

    # Return the test sets for future use
    return x_test, y_test
