#CNN classifier using keras.

import tensorflow.keras as kr
from argparse import ArgumentParser
from sys import argv
import numpy as np
from deepscribe.models.baselines import mlp_classifier

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--npz", help=".npz file with labeled images.")
    parser.add_argument("--tensorboard", help="tensorboard log file")
    parser.add_argument("--nlayers", help="number of hidden layers", type=int)
    parser.add_argument("--lsize", help="hidden layer size", type=int)
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

    input_shape = imgs[0].shape

    model = mlp_classifier(input_shape, args.nlayers, args.lsize, labels.shape[1])

    #compile with optimizer and loss function

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # tensorboard callbacks
    tensorboard = kr.callbacks.TensorBoard(args.tensorboard, histogram_freq=1, write_graph=1, write_grads=1)

    model.fit(imgs,
            labels,
            batch_size=args.bsize,
            epochs=args.epochs,
            verbose=1,
            callbacks = [tensorboard],
            validation_split=(1 - args.split))

if __name__=="__main__":
    run(parse(argv[1:]))
