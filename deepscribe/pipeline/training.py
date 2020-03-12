# model training luigi tasks
#

import luigi
from deepscribe.pipeline.selection import SelectDatasetTask
from deepscribe.models.baselines import cnn_classifier_2conv, cnn_classifier_4conv
import numpy as np
import json
from pathlib import Path


class TrainKerasModelFromDefinitionTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs
    sigma = luigi.FloatParameter(default=0.5)
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
            self.rest_as_other,
            self.whiten,
            self.epsilon,
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

        if "conv4_kernels" in model_params:
            _, model = cnn_classifier_4conv(
                data["train_imgs"],
                data["train_labels"],  # using sparse categorical cross-entropy
                data["valid_imgs"],
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
