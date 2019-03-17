#CNN classifier using keras.

import tensorflow.keras as kr
from argparse import ArgumentParser
from sys import argv
import numpy as np
from deepscribe.models.baselines import build_cnn_classifier
import socket
from datetime import datetime
import os
import talos
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.activations import relu, elu, softmax, sigmoid
from tensorflow.keras.losses import logcosh, binary_crossentropy, categorical_crossentropy

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--npz", help=".npz file with labeled images.")
    parser.add_argument("--tensorboard", help="tensorboard log file")
    parser.add_argument("--split", type=float, help="train/test split")
    parser.add_argument("--output", help="folder with log files and output plots")
    return parser.parse_args(args)


def train_model(x_train, y_train, x_val, y_val, params, tensorboard_dir=None):

    model = build_cnn_classifier(x_train[0].shape, y_train.shape[1], params)

    model.compile(optimizer=params['optimizer'](lr=talos.metrics.keras_metrics.lr_normalizer(params['lr'],params['optimizer'])),
                    loss='categorical_crossentropy',
                    metrics=['acc', 'fmeasure_acc'])

    if tensorboard_dir:
        # create tensorboard directory
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            tensorboard_dir, current_time + '_' + socket.gethostname())

        # tensorboard callbacks
        tensorboard = kr.callbacks.TensorBoard(log_dir, histogram_freq=1, write_graph=1, write_grads=1)
        #
        callbacks = [tensorboard]
    else:
        callbacks = []

    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        callbacks=callbacks,
                        verbose=0)

    return history, model


# define params dictionary
def params_dictionary():
    p = {'lr': (0.5, 5, 10),
         'conv1_kernels': [4, 8, 16, 32, 64],
         'conv1_ksize': [5, 10, 15, 20, 25],
         'conv1_stride': [1, 2, 3],
         'pool1_size': [1, 3, 5],
         'pool1_stride': [1, 3, 5],
         'conv2_kernels': [4, 8, 16, 32, 64],
         'conv2_ksize': [2, 3, 4],
         'conv2_stride': [1, 2, 3],
         'pool2_size': [1, 3, 5],
         'pool2_stride': [1, 3, 5],
         'dense_size': [64, 128, 256, 512],
         'epochs': [100, 150, 200],
         'dropout': (0, 0.4, 0.5, 0.7, 0.9),
         'optimizer': [Adam, Nadam, RMSprop],
         'activation': [relu, elu]}

    return p

def run(args):

    #load datasets
    loaded = np.load(args.npz)

    imgs = loaded['imgs']
    # convert labels to categorical (one-hot) labels
    labels = kr.utils.to_categorical(loaded['labels'])

    imgs, labels = shuffle(imgs, labels)
    print("starting Talos scan.")
    results = talos.Scan(imgs, labels,
    params=params_dictionary(),
    model=train_model,
    grid_downsample=.00001,
    debug=True,
    print_params=True)

    print(results)


if __name__=="__main__":
    run(parse(argv[1:]))
