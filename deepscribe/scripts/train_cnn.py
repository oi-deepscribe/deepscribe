#CNN classifier using keras.

import tensorflow.keras as kr
from argparse import ArgumentParser
from sys import argv
import numpy as np
from deepscribe.models.baselines import cnn_classifier
import socket
from datetime import datetime
import os
from sklearn.utils import shuffle

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--npz", help=".npz file with labeled images.")
    parser.add_argument("--tensorboard", help="tensorboard log file")
    parser.add_argument("--split", help="train/test split", type=float)
    parser.add_argument("--bsize", help="batch size for training", type=int)
    parser.add_argument("--epochs", help="number of training epochs.", type=int)
    parser.add_argument("--output", help="folder with log files and output plots")
    return parser.parse_args(args)


def run(args):

    #load datasets

    loaded = np.load(args.npz)

    imgs = loaded['imgs']
    # convert labels to categorical (one-hot) labels
    labels = kr.utils.to_categorical(loaded['labels'])

    imgs, labels = shuffle(imgs, labels)

    input_shape = imgs[0].shape

    model = cnn_classifier(input_shape, labels.shape[1])

    #compile with optimizer and loss function

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # create tensorboard directory
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        args.tensorboard, current_time + '_' + socket.gethostname())

    # tensorboard callbacks
    tensorboard = kr.callbacks.TensorBoard(log_dir, histogram_freq=1, write_graph=1, write_grads=1)
    #
    model.fit(imgs,
            labels,
            batch_size=args.bsize,
            epochs=args.epochs,
            verbose=2,
            callbacks = [tensorboard],
            validation_split=(1 - args.split))

if __name__=="__main__":
    run(parse(argv[1:]))
