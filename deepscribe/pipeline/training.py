# model training luigi tasks
#

import luigi
import tensorflow.keras as kr
import os
from deepscribe.pipeline.selection import SelectDatasetTask
from deepscribe.models.baselines import cnn_classifier_2conv
import numpy as np
import json
from pathlib import Path


# needed to get Talos to not freak out
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import talos

# TODO: merge this model training class with the talos definitions class and scikit-learn definitions with an abstract
# class
class TrainKerasModelFromDefinitionTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs
    num_augment = luigi.IntParameter(default=0)
    sigma = luigi.FloatParameter(default=0.5)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return SelectDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
            self.num_augment,
            self.sigma,
            self.rest_as_other,
        )

    def run(self):

        self.output().makedirs()

        # load model definition
        with open(self.model_definition, "r") as modelf:
            model_params = json.load(modelf)

        # update the params dict with number of classes

        model_params["num_classes"] = (
            len(self.keep_categories) + 1
            if self.rest_as_other
            else len(self.keep_categories)
        )

        # load data
        #
        data = np.load(self.input().path)

        # converting to one-hot

        _, model = cnn_classifier_2conv(
            data["train_imgs"],
            kr.utils.to_categorical(data["train_labels"]),
            data["valid_imgs"],
            kr.utils.to_categorical(data["valid_labels"]),
            model_params,
            data["classes"],
        )

        # save model for serialization
        model.save(self.output().path)

    def output(self):
        p = Path(self.model_definition)
        p_data = Path(self.input().path)

        return luigi.LocalTarget(
            "{}/{}_{}/trained.h5".format(self.modelsfolder, p.stem, p_data.stem)
        )


class RunTalosScanTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    talos_params = luigi.Parameter()  # JSON file with model definition specs
    nepoch = luigi.IntParameter(default=64)
    subsample = luigi.FloatParameter(default=1.0)
    num_augment = luigi.IntParameter(default=0)
    sigma = luigi.FloatParameter(default=0.5)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return SelectDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
            self.num_augment,
            self.sigma,
            self.rest_as_other,
        )

    def run(self):

        self.output().makedirs()

        # load talos parameters
        with open(self.talos_params, "r") as modelf:
            talos_params = json.load(modelf)

        # set the number of classes here

        p = Path(self.talos_params)

        # load data
        talos_params["num_classes"] = [
            len(self.keep_categories) + 1
            if self.rest_as_other
            else len(self.keep_categories)
        ]

        # adding the number of epochs as a command line argument
        talos_params["epochs"] = [self.nepoch]

        # load data
        #
        data = np.load(self.input().path)

        scan_object = talos.Scan(
            data["train_imgs"],
            kr.utils.to_categorical(data["train_labels"]),
            x_val=data["valid_imgs"],
            y_val=kr.utils.to_categorical(data["valid_labels"]),
            model=cnn_classifier_2conv,  # TODO: update this with new type signature
            params=talos_params,
            fraction_limit=self.subsample,
            experiment_name=p.stem,
        )

        # save DataFrame as CSV

        scan_object.data.to_pickle(self.output().path)

    def output(self):

        p = Path(self.talos_params)

        return luigi.LocalTarget(
            "{}/talos/{}_talos_{}_epoch_subsampled_{}.pkl".format(
                self.modelsfolder, p.stem, self.nepoch, self.subsample
            )
        )


# abstract task, overriden
