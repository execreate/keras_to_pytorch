import torch.nn as nn
from keras import Sequential, layers
import collections

from .utils import conv_layer_name, dense_layer_name, padding_to_number


class ModelConstructor:
    """
    Model Constructor
    Captures all hyper-parameters and builds up keras and pytorch models.
    """
    def __init__(self, input_shape, conv_layers, dense_layers, data_format="channels_first"):
        """
        Captures hyper-parameters
        :param input_shape: a tuple of integers. Example: (1, 96, 96).
        :param conv_layers: an array of mappings of conv layer parameters. Example: [{
                     "filters": 32,  # any integer
                     "kernel_size": (3, 3),  # or just 3
                     "strides": (1, 1),  # or just 1
                     "padding": "valid",  # or "same"
                     "activation": "relu",  # or "tanh" or "elu"
                     "max_pooling": {
                         "size": (2, 2),  # tuple of integers
                         "strides": (2, 2),  # tuple of integers
                     },
                     "dropout": 0.5,  # any real number in range (0, 1)
                     "kernel_initializer": "glorot_uniform",  # refer to https://keras.io/api/layers/initializers/
                 }, ...]
        :param dense_layers: an array of dense layer parameters. Example: [{
                     "input_units": 6400,  # any integer, you must set it for the first dense layer
                     "units": 1000,  # any integer
                     "dropout": 0.5,  # any real number in range (0, 1)
                     "activation": "relu",  # or "tanh" or "elu"
                     "kernel_initializer": "glorot_uniform",  # refer to https://keras.io/api/layers/initializers/
                 }, ...]
        :param data_format: data format string "channels_first" or "channels_last".
            Note that pytorch uses "channels_first", while for keras the default format is "channels_first".
        """
        assert type(input_shape) == tuple, "expected input_shape parameter to be a tuple"
        assert type(conv_layers) == list, "expected conv_layers parameter to be a list"
        assert type(dense_layers) == list, "expected dense_layers parameter to be a list"

        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.data_format = data_format

    def train_keras_model(self,
                          x_train, y_train,
                          validation_data,
                          train_callbacks,  # refer to https://keras.io/api/callbacks/
                          batch_size=128,
                          epochs=100,
                          optimizer="adam",  # refer to https://keras.io/api/optimizers/
                          loss="mean_squared_error",  # refer to https://keras.io/api/losses/
                          metrics="root_mean_squared_error",  # refer to https://keras.io/api/metrics/
                          verbose=True):
        keras_model = self.get_keras_model(verbose=verbose)
        keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        keras_model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        validation_data=validation_data, callbacks=train_callbacks)

        return keras_model

    def get_keras_model(self, verbose=False):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=self.input_shape))

        conv_counter = 0
        for conv_layer in self.conv_layers:
            conv_counter += 1
            kernel_initializer = conv_layer[
                "kernel_initializer"] if "kernel_initializer" in conv_layer else "glorot_uniform"

            model.add(layers.Conv2D(
                conv_layer["filters"],
                conv_layer["kernel_size"],
                strides=conv_layer["strides"],
                padding=conv_layer["padding"],
                activation=conv_layer["activation"],
                kernel_initializer=kernel_initializer,
                name=conv_layer_name(conv_counter),
                data_format=self.data_format
            ))
            model.add(layers.MaxPool2D(
                pool_size=conv_layer["max_pooling"]["size"],
                strides=conv_layer["max_pooling"].get("strides", None),
                data_format=self.data_format
            ))
            if "dropout" in conv_layer:
                model.add(layers.SpatialDropout2D(conv_layer["dropout"], data_format='channels_first'))

        model.add(layers.Flatten(data_format='channels_first'))

        dense_counter = 0
        for dense_layer in self.dense_layers:
            dense_counter += 1
            kernel_initializer = dense_layer[
                "kernel_initializer"] if "kernel_initializer" in dense_layer else "glorot_uniform"

            model.add(layers.Dense(
                dense_layer["units"],
                activation=dense_layer["activation"],
                kernel_initializer=kernel_initializer,
                name=dense_layer_name(dense_counter)
            ))

            if "dropout" in dense_layer:
                model.add(layers.Dropout(dense_layer["dropout"]))

        if verbose:
            model.summary()

        return model

    def get_pytorch_model(self, verbose=False):
        pt_conv_layers = []
        for idx, conv_layer in enumerate(self.conv_layers):
            if idx == 0:
                if self.data_format == "channels_first":
                    prev_layer_filters = self.input_shape[0]
                else:
                    prev_layer_filters = self.input_shape[-1]
            else:
                prev_layer_filters = self.conv_layers[idx - 1]["filters"]

            pt_conv_layers.append((
                conv_layer_name(idx + 1),
                nn.Conv2d(
                    prev_layer_filters,  # in_channels
                    conv_layer["filters"],  # out_channels
                    conv_layer["kernel_size"],
                    stride=conv_layer["strides"],
                    padding=padding_to_number(conv_layer["padding"], conv_layer["kernel_size"])
                )
            ))

            if conv_layer["activation"] == "relu":
                pt_conv_layers.append((conv_layer_name(idx + 1) + "_activation", nn.ReLU()))
            elif conv_layer["activation"] == "elu":
                pt_conv_layers.append((conv_layer_name(idx + 1) + "_activation", nn.ELU()))
            else:
                pt_conv_layers.append((conv_layer_name(idx + 1) + "_activation", nn.Tanh()))

            pt_conv_layers.append((conv_layer_name(idx + 1) + "_pooling",
                                   nn.MaxPool2d(conv_layer["max_pooling"]["size"],
                                                stride=conv_layer["max_pooling"].get("strides", None))
                                   ))

            if "dropout" in conv_layer:
                pt_conv_layers.append((conv_layer_name(idx + 1) + "_dropout",
                                       nn.Dropout2d(conv_layer["dropout"])))

        pt_dense_layers = []
        for idx, dense_layer in enumerate(self.dense_layers):
            if idx == 0:
                prev_layer_units = dense_layer["input_units"]
            else:
                prev_layer_units = self.dense_layers[idx - 1]["units"]

            pt_dense_layers.append((
                dense_layer_name(idx + 1),
                nn.Linear(
                    prev_layer_units,
                    dense_layer["units"]
                )
            ))

            if dense_layer["activation"] == "relu":
                pt_dense_layers.append((dense_layer_name(idx + 1) + "_activation", nn.ReLU()))
            elif dense_layer["activation"] == "elu":
                pt_dense_layers.append((dense_layer_name(idx + 1) + "_activation", nn.ELU()))
            else:
                pt_dense_layers.append((dense_layer_name(idx + 1) + "_activation", nn.Tanh()))

            if "dropout" in dense_layer:
                self.dense_layers.append((dense_layer_name(idx + 1) + "_dropout",
                                          nn.Dropout(dense_layer["dropout"])))

        model = nn.Sequential(collections.OrderedDict([
            *pt_conv_layers,
            ("flatten", nn.Flatten()),
            *pt_dense_layers
        ]))
        if verbose:
            print(model)

        return model
