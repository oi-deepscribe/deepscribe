# model training luigi tasks
#

import luigi
import tensorflow.keras as kr
import os
from deepscribe.luigi.ml_input import AssignDatasetTask
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainModelTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    epochs = luigi.IntParameter()
    batch_size = luigi.IntParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction

    def requires(self):
        return AssignDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
        )

    # TODO: load this from an external model definition
    def build_model(self):
        model = kr.models.Sequential()

        model.add(
            kr.layers.Conv2D(
                64,
                kernel_size=(16, 16),
                strides=(1, 1),
                activation="relu",
                input_shape=(self.target_size, self.target_size, 1),
            )
        )
        model.add(kr.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
        model.add(kr.layers.BatchNormalization())
        model.add(kr.layers.Dropout(0.4))
        model.add(
            kr.layers.Conv2D(32, kernel_size=(8, 8), strides=(1, 1), activation="relu")
        )
        model.add(kr.layers.BatchNormalization())
        model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(kr.layers.Dropout(0.6))
        model.add(kr.layers.Flatten())
        model.add(kr.layers.Dense(512, activation="relu"))
        model.add(kr.layers.Dense(len(self.keep_categories), activation="softmax"))

        return model

    def run(self):
        # build model
        model = self.build_model()

        # TODO: set learning rate
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["acc"]
        )

        # load data
        #
        data = np.load(self.input().path)

        # converting to one-hot

        # train model!
        # TODO: early stopping
        history = model.fit(
            data["train_imgs"],
            kr.utils.to_categorical(data["train_labels"]),
            batch_size=32,
            epochs=5,
            validation_data=(
                data["valid_imgs"],
                kr.utils.to_categorical(data["valid_labels"]),
            ),
        )

        # save model for serialization
        model.save(self.output().path)

    def output(self):
        return luigi.LocalTarget(
            "{}_{}_epochs_model.h5".format(
                os.path.splitext(self.input().path)[0], self.epochs
            )
        )


class TestModelTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    epochs = luigi.IntParameter()
    batch_size = luigi.IntParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction

    def requires(self):
        return {
            "model": TrainModelTask(
                self.imgfolder,
                self.hdffolder,
                self.target_size,
                self.keep_categories,
                self.epochs,
                self.batch_size,
                self.fractions,
            ),
            "dataset": AssignDatasetTask(
                self.imgfolder,
                self.hdffolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
            ),
        }

    def run(self):
        # load TF model and dataset
        model = kr.models.load_model(self.input()["model"].path)
        data = np.load(self.input()["dataset"].path)

        # make predictions on data

        # TODO: determine format of model.predict
        pred_labels = model.predict(data["test_imgs"])

        # compute confusion matrix

        confusion = confusion_matrix(data["test_labels"], pred_labels)

        np.save(self.output().path, confusion)

    def output(self):
        return luigi.LocalTarget(
            "{}_confusion.npy".format(os.path.splitext(self.input()["dataset"].path)[0])
        )


class PlotConfusionMatrixTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    epochs = luigi.IntParameter()
    batch_size = luigi.IntParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction

    def requires(self):
        return TestModelTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.epochs,
            self.batch_size,
            self.fractions,
        )

    def run(self):
        # load matrix

        confusion = np.load(self.input().path)

        plt.figure()
        plt.title("Confusion matrix from {}".format(self.input().path))
        plt.matshow(confusion)
        plt.savefig(self.output().path)

    def output(self):
        return luigi.LocalTarget(
            "{}_confusion.png".format(os.path.splitext(self.input().path)[0])
        )
