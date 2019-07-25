# model training luigi tasks
#
#
import luigi
import tensorflow.keras as kr
import os
from deepscribe.luigi.ml_input import AssignDatasetTask
from tensorflow.keras.activations import relu, elu, softmax, sigmoid
import numpy as np

class TrainModelTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter() #standardizing to square images
    keep_categories = luigi.ListParameter()
    epochs = luigi.IntParameter()
    #TODO: this as param
    fractions = [0.7, 0.1, 0.2] # train/valid/test fraction

    def requires(self):
        return AssignDatasetTask(self.imgfolder,
                                 self.hdffolder,
                                 self.target_size,
                                 self.keep_categories)


    def build_model(self):
        model = kr.models.Sequential()

        model.add(kr.layers.Conv2D(64,
                                   kernel_size=(16, 16),
                                   strides=(1, 1),
                         activation=relu,
                         input_shape=(self.target_size, self.target_size, 1)))
        model.add(kr.layers.MaxPooling2D(pool_size=(3, 3),
                                        strides=(1, 1)))
        model.add(kr.layers.BatchNormalization())
        model.add(kr.layers.Dropout(0.4))
        model.add(kr.layers.Conv2D(32,
                kernel_size=(8, 8),
                strides=(1, 1),
                activation=relu))
        model.add(kr.layers.BatchNormalization())
        model.add(kr.layers.MaxPooling2D(pool_size=(2, 2),
                                            strides=(1, 1)))

        model.add(kr.layers.Dropout(0.6))
        model.add(kr.layers.Flatten())
        model.add(kr.layers.Dense(512, activation=relu))
        # model.add(kr.layers.Dense(512, activation='relu'))
        model.add(kr.layers.Dense(len(self.keep_categories), activation='softmax'))

        return model



    def run(self):
        # build model
        model = self.build_model()

        # TODO: set learning rate
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['acc'])

        # load data
        #
        data = np.load(self.input().path)

        # converting to one-hot

        #train model!
        #TODO: early stopping
        history = model.fit(data['train_imgs'],
                            kr.utils.to_categorical(data['train_labels']),
                            batch_size=32,
                            epochs=5,
                            validation_data=(data['valid_imgs'], kr.utils.to_categorical(data['valid_labels'])))

        # save model for serialization
        model.save(self.output().path)

    def output(self):
        return luigi.LocalTarget("{}_{}_epochs_model.h5".format(os.path.splitext(self.input().path)[0], self.epochs))


class TestModelTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter() #standardizing to square images
    keep_categories = luigi.ListParameter()
    #TODO: this as param
    fractions = [0.7, 0.1, 0.2] # train/valid/test fraction

    def requires(self):
        return {'model': TrainModelTask(self.imgfolder,
                              self.hdffolder,
                              self.target_size,
                              self.keep_categories),
                'dataset': AssignDatasetTask(self.imgfolder,
                                         self.hdffolder,
                                         self.target_size,
                                         self.keep_categories)}


    def output(self):
        return luigi.LocalTarget("{}_confusion.npz".format(os.path.splitext(self.input().path)[0]))
