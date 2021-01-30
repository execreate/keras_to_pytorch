# Train with Keras, evaluate with PyTorch

I came up with the code in this repository while doing my homeworks on
[Introduction to Deep Learning class](https://campus.tum.de/tumonline/WBMODHB.wbShowMHBReadOnly?pKnotenNr=1334293).
At the moment the assignments in this class are based on [PyTorch](https://pytorch.org) machine learning framework
which does not support GPU acceleration on macOS with AMD GPUs. In order to utilize the hardware at my disposal
and significantly improve the training time, while playing around with neural networks to solve my assignments,
I started training my networks using [Keras](https://keras.io) with [PlaidML](https://github.com/plaidml/plaidml)
backend. At the end of the day I had to convert my Keras model into a PyTorch model in order to submit
my solution for evaluation.

>! It did not work :(
>! Refer to the notebook in the repository root for more details.

## Getting started

Create a virtual environment with Python 3.8 using [virtualenv](https://docs.python.org/3.8/library/venv.html)
or [anaconda](https://docs.anaconda.com/anaconda/install/) and install the requirements:
```
pip install -r requirements.txt
```

Also make sure to set the default device for PladML:
```
plaidml-setup
```

Start a jupyter notebook server (just run `jupyter notebook` in your console) and feel free to play around with
[plug_n_play.ipynb](https://github.com/execreate/keras_to_pytorch/blob/master/plug_n_play.ipynb).

## Going deeper

You can also use [Tensorflow](https://www.tensorflow.org) backend for Keras instead of PlaidML, just make sure
to change the imports.

Take a look on
[models/model_constructor.py](https://github.com/execreate/keras_to_pytorch/blob/master/models/model_constructor.py)
to see how Keras and PyTorch models are constructed. In
[models/utils.py](https://github.com/execreate/keras_to_pytorch/blob/master/models/utils.py) you'll find the code,
which copies the weights from Keras model to a PyTorch model.
