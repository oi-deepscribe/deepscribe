# model training luigi tasks
#

import luigi
import tensorflow.keras as kr
import os
from deepscribe.luigi.ml_input import AssignDatasetTask
from deepscribe.models.baselines import cnn_classifier
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import talos
from pathlib import Path


class TrainModelFromDefinitionTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
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

        # update the params dict with number of classes

        model_params["num_classes"] = len(self.keep_categories)

        # load data
        #
        data = np.load(self.input().path)

        # converting to one-hot

        # convert to correct tensor size

        if len(data["train_imgs"].shape) < 4:
            train_data = np.expand_dims(data["train_imgs"], axis=-1)
        else:
            train_data = data["train_imgs"]

        model = cnn_classifier(
            train_data,
            kr.utils.to_categorical(data["train_labels"]),
            data["valid_imgs"],
            kr.utils.to_categorical(data["valid_labels"]),
            model_params,
        )

        # save model for serialization
        model.save(self.output().path)

    def output(self):

        p = Path(self.model_definition)

        return luigi.LocalTarget("{}/{}_trained.h5".format(self.modelsfolder, p.stem))


class RunTalosScanTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    talos_params = luigi.Parameter()  # JSON file with model definition specs
    subsample = luigi.FloatParameter()

    def requires(self):
        return AssignDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
        )

    def run(self):

        # load talos parameters
        with open(self.talos_params, "r") as modelf:
            talos_params = json.load(modelf)

        # set the number of classes here
        talos_params["num_classes"] = [len(self.keep_categories)]

        p = Path(self.talos_params)

        # load data
        data = np.load(self.input().path)

        # converting to one-hot

        # convert to correct tensor size

        if len(data["train_imgs"].shape) < 4:
            train_data = np.expand_dims(data["train_imgs"], axis=-1)
        else:
            train_data = data["train_imgs"]

        scan_object = talos.Scan(
            train_data,
            kr.utils.to_categorical(data["train_labels"]),
            x_val=data["valid_imgs"],
            y_val=kr.utils.to_categorical(data["valid_labels"]),
            model=cnn_classifier,
            params=talos_params,
            fraction_limit=self.subsample,
            experiment_name=p.stem,
        )

        # serialize scan object and save

        # TODO: check if serialization works

        with self.output().open("w") as outf:
            json.dump(scan_object, outf)

    def output(self):

        p = Path(self.talos_params)

        return luigi.LocalTarget(
            "{}/{}_talos_subsampled_{}.json".format(
                self.modelsfolder, p.stem, self.subsample
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
