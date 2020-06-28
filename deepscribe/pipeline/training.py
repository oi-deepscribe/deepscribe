# model training luigi tasks
#

import luigi
from deepscribe.pipeline.selection import SelectDatasetTask
from luigi.util import requires
from deepscribe.models.train import build_train_params
from deepscribe.models.baselines import cnn_classifier_2conv, cnn_classifier_4conv
from deepscribe.models.cnn import VGG16, VGG19, ResNet50, ResNet50V2, ResNet18
import numpy as np
import json
from pathlib import Path
from abc import ABC
import os

# needed to get Talos to not freak out
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import talos


@requires(SelectDatasetTask)
class TrainKerasModelTask(luigi.Task):
    """
    Parametrize the model entirely from Luigi parameters instead of using a separate params file.
    Produces longer file names and longer task specification, but ultimately makes it easier to iterate on experiments. 

    Parameters are used to build a dictionary that is passed to a model building class, for compatibility with Talos.
    """

    modelsfolder = luigi.Parameter()
    epochs = luigi.IntParameter()
    bsize = luigi.IntParameter()
    architecture = luigi.Parameter(default="resnet18")  # architecture ID
    threshold = luigi.BoolParameter(default=False)
    reweight = luigi.BoolParameter(default=False)
    optimizer = luigi.Parameter(default="adam")
    lr = luigi.FloatParameter(default=0.001)  # learning rate
    activation = luigi.Parameter(default="relu")
    early_stopping = luigi.IntParameter(default=0)
    reduce_lr = luigi.IntParameter(default=0)
    oversample = luigi.BoolParameter(default=False)
    shear = luigi.FloatParameter(default=0.0)
    zoom = luigi.FloatParameter(default=0.0)
    width_shift = luigi.FloatParameter(default=0.0)
    height_shift = luigi.FloatParameter(default=0.0)
    rotation_range = luigi.FloatParameter(default=0.0)
    l1 = luigi.FloatParameter(default=0.0, description="penalty for l1 regularization")
    l2 = luigi.FloatParameter(default=0.0, description="penalty for l2 regularization")
    focal = luigi.FloatParameter(default=0.0, description="focal loss parameter")
    momentum = luigi.FloatParameter(
        default=0.0, description="momentum parameter for SGD."
    )
    shadow_val_range = luigi.ListParameter(default=[0.0, 0.0])
    num_shadows = luigi.IntParameter(default=0)
    input_dropout = luigi.FloatParameter(default=0.0)

    def run(self):

        # create output directory
        self.output().makedirs()
        # assemble parameter dictionary from luigi args
        model_params = {
            name: self.__getattribute__(name) for name, _ in self.get_params()
        }

        data = np.load(self.input().path)
        # get correct number of classes for model building
        model_params["num_classes"] = len(data["classes"])

        _, model = build_train_params(
            data["train_imgs"],
            data["train_labels"],  # using sparse categorical cross-entropy
            data["valid_imgs"],
            data["valid_labels"],
            model_params,
        )

        # save model for serialization
        model.save(self.output().path)

    def output(self):

        """

        Output location of trained Keras model in HDF5 format.

        :return: luigi.LocalTarget
        """
        # TODO: refactor this using requires decorator
        # filtering out parameters that are already encoded in file name
        training_params = [
            param
            for param, obj in self.get_params()
            if param
            not in [
                "imgarchive",
                "target_size",
                "histogram",
                "sigma",
                "keep_categories",
                "fractions",
                "rest_as_other",
                "whiten",
                "epsilon",
            ]
        ]

        training_param_vals = [
            str(self.__getattribute__(param)) for param in training_params
        ]

        return luigi.LocalTarget(
            "{}/{}_{}/trained.h5".format(
                self.modelsfolder,
                Path(self.input().path).stem,
                "_".join(training_param_vals).replace(".", "p"),
            )
        )


@requires(SelectDatasetTask)
class RunTalosScanTask(luigi.Task):
    """

    Runs a Talos scan from the model_definition parameter (a dictionary of lists instead of single values) 

    """

    model_definition = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    subsample = luigi.FloatParameter(default=0.001)

    def run(self):

        self.output().makedirs()

        with open(self.model_definition, "r") as modelf:
            talos_params = json.load(modelf)

        # set the number of classes here
        data = np.load(self.input().path)
        # gotta add it as a list for Talos...
        talos_params["num_classes"] = [len(data["classes"])]

        scan_object = talos.Scan(
            data["train_imgs"],
            data["train_labels"],
            x_val=data["valid_imgs"],
            y_val=data["valid_labels"],
            model=build_train_params,
            params=talos_params,
            fraction_limit=self.subsample,
            experiment_name=Path(self.model_definition).stem,
        )

        # save DataFrame as CSV

        scan_object.data.to_csv(self.output().path)

    def output(self):
        """

        :return: luigi.LocalTarget
        """
        return luigi.LocalTarget(
            "{}/{}_{}/talos_scan_subsample_{}.csv".format(
                self.modelsfolder,
                Path(self.input().path).stem,
                Path(self.model_definition).stem,
                str(self.subsample).replace(".", "_"),
            )
        )
