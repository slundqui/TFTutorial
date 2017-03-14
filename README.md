# TFTutorial
Code for a tutorial for TensorFlow

Prerequisites:
- Python 3
- Tensorflow 1.0

slp.py is the tutorial script to run. This will make 2 directories:

MNIST_data: The directory to the input MNIST digits

output: The output directory of the script for tensorboard

To run tensorboard, navigate to the directory containing the output directory and run the following command:

```bash
tensorboard --logdir=output --port=1111
```

To view, open a web browser and type the following address into the address bar:

<localhost:1111>

