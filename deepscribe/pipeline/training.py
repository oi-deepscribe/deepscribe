# model training luigi tasks
#

import luigi
from deepscribe.pipeline.selection import SelectDatasetTask
from deepscribe.models.baselines import cnn_classifier_2conv, cnn_classifier_4conv
from deepscribe.models.cnn import VGG16, VGG19, ResNet50, ResNet50V2
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


class TrainModelFromDefinitionTask(luigi.Task, ABC):
    """

    Luigi task skeleton for a task that loads parameters from a JSON file, trains a model based on those parameters,
    and saves the model (or other results) to disk. 

    """

    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs
    sigma = luigi.FloatParameter(default=0.5)
    threshold = luigi.BoolParameter(default=False)

    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths
    whiten = luigi.BoolParameter(default=False)
    epsilon = luigi.FloatParameter(default=0.1)

    def requires(self):
        return SelectDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
            self.sigma,
            self.threshold,
            self.rest_as_other,
            self.whiten,
            self.epsilon,
        )

    def load_def(self):
        """

        Loads and preprocesses the model definition JSON file.

        """
        # loads model definition
        raise NotImplementedError

    def run_training(self, model_params: dict):
        """

        Executes model training and saves result to disk.

        :param model_params: dictionary containing model parameter information.
        """
        raise NotImplementedError

    def run(self):

        self.output().makedirs()

        model_def = self.load_def()

        self.run_training(model_def)


class TrainKerasModelFromDefinitionTask(TrainModelFromDefinitionTask):
    """

    Trains a Keras model from the model_definition parameter and saves it to disk. 

    """

    def load_def(self):
        with open(self.model_definition, "r") as modelf:
            model_params = json.load(modelf)

        # update the params dict with number of classes

        model_params["num_classes"] = (
            len(self.keep_categories) + 1
            if self.rest_as_other
            else len(self.keep_categories)
        )

        model_params["SLURM_RUN"] = os.environ.get("SLURM_JOB_ID", "NONE")

        return model_params

    def run_training(self, model_params: dict):
        data = np.load(self.input().path)

        if "conv4_kernels" in model_params:
            _, model = cnn_classifier_4conv(
                data["train_imgs"],
                data["train_labels"],  # using sparse categorical cross-entropy
                data["valid_imgs"],
                data["valid_labels"],
                model_params,
            )
        elif "architecture" in model_params and model_params["architecture"] == "vgg16":

            # TODO: build image dimension handling into the model object?

            _, model = VGG16()(
                np.repeat(data["train_imgs"], 3, axis=3),  # vgg16 expects RGB
                data["train_labels"],  # using sparse categorical cross-entropy
                np.repeat(data["valid_imgs"], 3, axis=3),
                data["valid_labels"],
                model_params,
            )
        elif "architecture" in model_params and model_params["architecture"] == "vgg19":

            # TODO: build image dimension handling into the model object?

            _, model = VGG19()(
                np.repeat(data["train_imgs"], 3, axis=3),  # vgg19 expects RGB
                data["train_labels"],  # using sparse categorical cross-entropy
                np.repeat(data["valid_imgs"], 3, axis=3),
                data["valid_labels"],
                model_params,
            )

        elif (
            "architecture" in model_params
            and model_params["architecture"] == "resnet50"
        ):
            _, model = ResNet50()(
                np.repeat(data["train_imgs"], 3, axis=3),  # resnet expects RGB
                data["train_labels"],  # using sparse categorical cross-entropy
                np.repeat(data["valid_imgs"], 3, axis=3),
                data["valid_labels"],
                model_params,
            )

        elif (
            "architecture" in model_params
            and model_params["architecture"] == "resnet50v2"
        ):
            _, model = ResNet50V2()(
                np.repeat(data["train_imgs"], 3, axis=3),  # resnet expects RGB
                data["train_labels"],  # using sparse categorical cross-entropy
                np.repeat(data["valid_imgs"], 3, axis=3),
                data["valid_labels"],
                model_params,
            )

        else:
            _, model = cnn_classifier_2conv(
                data["train_imgs"],
                data["train_labels"],  # using sparse categorical cross-entropy
                data["valid_imgs"],
                data["valid_labels"],
                model_params,
            )
        # save model for serialization
        model.save(self.output().path)

    def output(self):
        p = Path(self.model_definition)
        p_data = Path(self.input().path)

        return luigi.LocalTarget(
            "{}/{}_{}/trained.h5".format(self.modelsfolder, p.stem, p_data.stem)
        )


class RunTalosScanTask(TrainModelFromDefinitionTask):
    """

    Runs a Talos scan from the model_definition parameter (a dictionary of lists instead of single values) 

    """

    subsample = luigi.FloatParameter(default=0.001)

    def load_def(self):
        with open(self.model_definition, "r") as modelf:
            talos_params = json.load(modelf)

        # set the number of classes here

        talos_params["num_classes"] = [
            len(self.keep_categories) + 1
            if self.rest_as_other
            else len(self.keep_categories)
        ]

        return talos_params

    def run_training(self, model_params: dict):
        data = np.load(self.input().path)

        scan_object = talos.Scan(
            data["train_imgs"],
            data["train_labels"],
            x_val=data["valid_imgs"],
            y_val=data["valid_labels"],
            model=cnn_classifier_2conv,
            params=model_params,
            fraction_limit=self.subsample,
            experiment_name=Path(self.model_definition).stem,
        )

        # save DataFrame as CSV

        scan_object.data.to_pickle(self.output().path)

    def output(self):
        p = Path(self.model_definition)
        p_data = Path(self.input().path)

        return luigi.LocalTarget(
            "{}/{}_{}/talos_scan.pkl".format(self.modelsfolder, p.stem, p_data.stem)
        )
