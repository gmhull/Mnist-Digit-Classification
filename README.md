# Mnist-Digit-Classification

This was a machine learning tutorial project using the mnist handwriting dataset.  

The homemade folder was designed as a way to create the neural network directly without using outside packages.  This uses the mnist dataset downloaded from http://yann.lecun.com/exdb/mnist/.  The dataset is not included in this repo.  

The tf folder includes the same files using the tensorflow library.  Neural network using tensorflow was inspired by the tutorial by setdex.  This one works a lot more efficiently.

There are two different types of files:
Training - Trains the model and then creates a window to show the image from the test dataset and the guess that the model predicted.
Drawing Digits - The user may draw on the screen and then use the model to see if the drawn image matches any digits.
