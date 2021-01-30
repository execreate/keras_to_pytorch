import torch
import torch.nn as pt_nn
import numpy as np


def conv_layer_name(num):
    return "conv2d_%d" % num


def dense_layer_name(num):
    return "dense_%d" % num


def padding_to_number(padding, kernel):
    if padding == "valid":
        return 0, 0

    if type(kernel) == tuple:
        h, w = kernel[0], kernel[1]
    else:
        h, w = kernel, kernel

    return int(h / 2), int(w / 2)


def get_keras_model_weights_dict(keras_model, pt_model, flip_channels=False):
    conv_counter = 0
    dense_counter = 0

    weight_dict = dict()
    for current_layer in pt_model:
        weights = None
        layer_name = None
        if isinstance(current_layer, pt_nn.Conv2d):
            conv_counter += 1
            layer_name = conv_layer_name(conv_counter)
            weights = keras_model.get_layer(name=layer_name).get_weights()
        elif isinstance(current_layer, pt_nn.Linear):
            dense_counter += 1
            layer_name = dense_layer_name(dense_counter)
            weights = keras_model.get_layer(name=layer_name).get_weights()

        # googled the problem, made some fixes to the code
        # but still does not work :(
        if weights is not None:
            w = weights[0]
            if len(w.shape) == 4:  # conv layer
                w = w.transpose(3, 2, 0, 1)
                if flip_channels:
                    w = w[::-1, ::-1]
                # flip filters
                w = w[..., ::-1, ::-1].copy()
            else:                  # dense layer
                w = w.transpose()
                if flip_channels:
                    w = w[::-1]
                # flip filters
                w = w[..., ::-1].copy()
            # print(w.shape)
            weight_dict['%s.weight' % layer_name] = w
            weight_dict['%s.bias' % layer_name] = weights[1].transpose()

    return weight_dict


def copy_weights_from_keras_to_pytorch(keras_model, pt_model):
    weight_dict = get_keras_model_weights_dict(keras_model, pt_model)
    model_state_dict = pt_model.state_dict()

    for k in model_state_dict:
        model_state_dict[k] = torch.from_numpy(weight_dict[k])

    pt_model.load_state_dict(model_state_dict)

    return pt_model


def weights_are_equal(keras_model, pt_model):
    weight_dict = get_keras_model_weights_dict(keras_model, pt_model)
    model_state_dict = pt_model.state_dict()

    for k in model_state_dict:
        if not np.array_equal(model_state_dict[k].numpy(), weight_dict[k]):
            print("Layer %s differs from keras model" % k)
        print("Weights on %s are equal to keras model" % k)


def get_scores(x, y, keras_model, pt_model, pt_loss=None, batch_size=128):
    keras_metrics = keras_model.evaluate(x, y, batch_size=batch_size)
    print("keras model loss: ", keras_metrics)
    print("keras model score: ", 1.0 / (2 * keras_metrics))

    if pt_loss is None:
        pt_loss = pt_nn.MSELoss()
    pt_model.eval()
    input_x = torch.from_numpy(x).squeeze().unsqueeze(1)
    pt_loss = pt_loss(pt_model(input_x), torch.from_numpy(y)).item()
    print("pytorch model loss: ", pt_loss)
    print("pytorch model score: ", 1.0 / (2 * pt_loss))
