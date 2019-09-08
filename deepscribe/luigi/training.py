# model training luigi tasks
#

import luigi
import tensorflow.keras as kr
import os
from deepscribe.luigi.ml_input import AssignDatasetTask
from deepscribe.models.baselines import build_cnn_classifier
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainModelFromDefinitionTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs

    def requires(self):
        return AssignDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
        )

    def run(self):

        # load model definition
        with open(self.model_definition, "r") as modelf:
            model_params = json.load(modelf)

        # build model
        model = build_cnn_classifier(
            (self.target_size, self.target_size, 1),
            len(self.keep_categories),
            model_params,
        )

        # TODO: set learning rate
        model.compile(
            optimizer=model_params["optimizer"],
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

        # load data
        #
        data = np.load(self.input().path)

        # converting to one-hot

        # convert to correct tensor size

        if len(data["train_imgs"].shape) < 4:
            train_data = np.expand_dims(data["train_imgs"], axis=-1)
        else:
            train_data = data["train_imgs"]

        print(train_data.shape)

        # train model!
        # TODO: early stopping
        history = model.fit(
            train_data,
            kr.utils.to_categorical(data["train_labels"]),
            batch_size=model_params["batch_size"],
            epochs=model_params["epochs"],
            validation_data=(
                data["valid_imgs"],
                kr.utils.to_categorical(data["valid_labels"]),
            ),
        )

        # save model for serialization
        model.save(self.output().path)

    def output(self):

        # load model definition - to set output params
        with open(self.model_definition, "r") as modelf:
            model_params = json.load(modelf)

        return luigi.LocalTarget(
            "{}_{}_epochs_{}_model.h5".format(
                os.path.splitext(self.input().path)[0],
                model_params["epochs"],
                os.path.basename(self.model_definition),
            )
        )


class TestModelTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs

    def requires(self):
        return {
            "model": TrainModelFromDefinitionTask(
                self.imgfolder,
                self.hdffolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
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
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs

    def requires(self):
        return TestModelTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
            self.model_definition,
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
