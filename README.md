# TFTutorial
Code for a tutorial for TensorFlow

Prerequisites:
Python 3
Tensorflow 1.0

slp.py: Tutorial script to run. This will make 2 folders:

MNIST_data: The directory to the input MNIST digits
output: The output directory of the script for tensorboard

To run tensorboard, navigate to the directory containing the output directory and run the following command:

tensorboard --logdir=output --port=1111

