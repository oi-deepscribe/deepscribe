#CNN classifier using keras.


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from data_utils import load_dataset
from argparse import ArgumentParser
from sys import argv
import numpy as np



def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--train_labels", help="location of CSV file with training labels")
    parser.add_argument("--train_images", help="location of training images ")
    parser.add_argument("--test_labels", help="location of CSV file with test labels")
    parser.add_argument("--test_images", help="location of test images ")
    parser.add_argument("-flatten", help="flatten images to grayscale.", action="store_true")
    parser.add_argument("--output", help="folder with log files and output plots")
    return parser.parse_args(args)

def build_model(input_shape, num_classes):
    '''
    Builds a CNN classifier with the provided input shape and number of classes.
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    #compile with optimizer and loss function
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def run(args):

    #load datasets

    train_x, train_y = load_dataset(args.train_labels, args.train_images, flatten=args.flatten)

    test_x, test_y = load_dataset(args.test_labels, args.test_images, flatten=args.flatten)

    input_shape = train_x[0].shape

    unique_classes = np.unique(np.concatenate((train_y, test_y)))

    character_dict = {character:i for i, character in enumerate(unique_classes)}

    train_y_categorical = np_utils.to_categorical([character_dict[char] for char in train_y])
    test_y_categorical = np_utils.to_categorical([character_dict[char] for char in test_y])
    cnn = build_model(input_shape, len(unique_classes))

    cnn.fit(train_x,
            train_y_categorical,
            batch_size=128,
            epochs=10,
            verbose=1,
            validation_data=(test_x, test_y_categorical))

    score = cnn.evaluate(test_x, test_y_categorical, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__=="__main__":
    run(parse(argv[1:]))
