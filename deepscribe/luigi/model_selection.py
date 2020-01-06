import luigi
from .training import TrainKerasModelFromDefinitionTask
from .ml_input import AssignDatasetTask
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import tensorflow.keras as kr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# produces confusion matrics from test data
class TestModelTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs
    num_augment = luigi.IntParameter(default=0)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return {
            "model": TrainKerasModelFromDefinitionTask(
                self.imgfolder,
                self.hdffolder,
                self.modelsfolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
                self.num_augment,
                self.rest_as_other,
            ),
            "dataset": AssignDatasetTask(
                self.imgfolder,
                self.hdffolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.num_augment,
                self.rest_as_other,
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
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs

    def requires(self):
        return TestModelTask(
            self.imgfolder,
            self.hdffolder,
            self.modelsfolder,
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


# selects the best architecture for this task, saves it to a JSON w/result vals.
# class SelectBestArchitectureTask(luigi.Task):
#     imgfolder = luigi.Parameter()
#     hdffolder = luigi.Parameter()
#     modelsfolder = luigi.Parameter()
#     target_size = luigi.IntParameter()  # standardizing to square images
#     keep_categories = luigi.ListParameter()
#     fractions = luigi.ListParameter()  # train/valid/test fraction
#     talos_params = luigi.Parameter()  # JSON file with model definition specs
#     subsample = luigi.FloatParameter()
