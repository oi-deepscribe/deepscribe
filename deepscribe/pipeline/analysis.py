import luigi
from .training import TrainKerasModelFromDefinitionTask
from .selection import SelectDatasetTask
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from abc import ABC
import os
import tensorflow.keras as kr
import tensorflow as tf
import matplotlib
from pathlib import Path
import json

matplotlib.use("Agg")
# import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt


class AnalysisTask(luigi.Task, ABC):
    """
    Abstract class that requires the completion of dataset selection and model training.

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
        """

        Task depends on a trained model and a dataset archive.

        :return: Dictionary with TrainKerasModelFromDefinitionTask and SelectDatasetTask
        """
        return {
            "model": TrainKerasModelFromDefinitionTask(
                self.imgfolder,
                self.hdffolder,
                self.modelsfolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
                self.sigma,
                self.threshold,
                self.rest_as_other,
                self.whiten,
                self.epsilon,
            ),
            "dataset": SelectDatasetTask(
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
            ),
        }


class PlotConfusionMatrixTask(AnalysisTask):
    """
    Produces a confusion matrix from the model on test data. 

    """

    def run(self):
        """
        Loads the model, runs it on test data, saves the confusion matrix as png.

        :return: None
        """
        # load matrix

        # load TF model and dataset
        model = kr.models.load_model(self.input()["model"].path)
        data = np.load(self.input()["dataset"].path)

        # make predictions on data

        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])

        # computing predicted labels
        pred_labels = np.argmax(pred_logits, axis=1)

        # compute confusion matrix

        confusion = confusion_matrix(data["test_labels"], pred_labels)

        # get labels from dataset

        data = np.load(self.input()["dataset"].path)

        class_labels = data["classes"]
        fig = plt.figure(figsize=(13, 13))
        ax = fig.add_subplot(111)
        plt.title("Confusion matrix")
        cax = ax.matshow(confusion)
        fig.colorbar(cax)

        ax.set_xticks(list(range(len(class_labels))))
        ax.set_yticks(list(range(len(class_labels))))
        # appending extra tick label to make sure everything aligns properly
        ax.set_xticklabels(list(class_labels))
        ax.set_yticklabels(list(class_labels))
        plt.savefig(self.output().path)

    def output(self):
        """

        Location of the output image.

        :return: luigi.LocalTarget
        """

        p = Path(self.model_definition)
        p_data = Path(self.input()["dataset"].path)

        return luigi.LocalTarget(
            "{}/{}_{}/confustion_test.png".format(
                self.modelsfolder, p.stem, p_data.stem
            )
        )


class GenerateClassificationReportTask(AnalysisTask):
    """

    Generates a classification report from the trained model on test data. 

    """

    def run(self):
        """
        Loads the model, runs it on test data, saves the classification report as txt.


        :return: None
        """
        # load TF model and dataset
        model = kr.models.load_model(self.input()["model"].path)
        data = np.load(self.input()["dataset"].path)

        # make predictions on data

        # TEST DATA

        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])

        # computing predicted labels
        pred_labels = np.argmax(pred_logits, axis=1)

        # compute confusion matrix

        report = classification_report(
            data["test_labels"], pred_labels, target_names=data["classes"]
        )

        # Compute the top-k accuracy across all of the data:

        top_k_test = []

        for i in range(2, min(5, len(data["classes"]))):
            k_i = kr.metrics.sparse_top_k_categorical_accuracy(
                data["test_labels"], pred_labels, k=i
            )

            top_k_test.append(f"top-{i} accuracy: {k_i}")

        # TRAIN DATA

        # (batch_size, num_classes)
        pred_logits = model.predict(data["train_imgs"])

        # computing predicted labels
        pred_labels = np.argmax(pred_logits, axis=1)

        # compute confusion matrix

        report_train = classification_report(
            data["train_labels"], pred_labels, target_names=data["classes"]
        )

        top_k_train = []

        # top-k accuracy on all data
        for i in range(2, min(5, len(data["classes"]))):
            k_i = kr.metrics.sparse_top_k_categorical_accuracy(
                data["train_labels"], pred_labels, k=i
            )

            top_k_train.append(f"top-{i} accuracy: {k_i}")

        with self.output().temporary_path() as temppath:
            with open(temppath, "w") as outf:
                outf.write("REPORT ON TEST\n")
                outf.write(report)
                outf.write("TOP-K ACCURACIES ON TEST")
                outf.write("\n".join(top_k_test))
                outf.write("REPORT ON TRAIN\n")
                outf.write(report_train)
                outf.write("TOP-K ACCURACIES ON TRAIN")
                outf.write("\n".join(top_k_train))

    def output(self):
        """

        Location of the output report.

        :return: luigi.LocalTarget
        """

        p = Path(self.model_definition)
        p_data = Path(self.input()["dataset"].path)

        return luigi.LocalTarget(
            "{}/{}_{}/classification_report.txt".format(
                self.modelsfolder, p.stem, p_data.stem
            )
        )


# plots a random sample of 16 incorrect images from test..
class PlotIncorrectTask(AnalysisTask):
    """

    Loads 16 incorrectly classified images from test data and plots them in a grid.

    """

    def run(self):
        """
        Loads model, runs on test data, picks 16 random incorrectly classified images.

        :return:
        """

        # load TF model and dataset
        model = kr.models.load_model(self.input()["model"].path)
        data = np.load(self.input()["dataset"].path)
        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])
        # (batch_size,)

        pred_labels = np.argmax(pred_logits, axis=1)

        incorrect_prediction_idx, = np.not_equal(
            data["test_labels"], pred_labels
        ).nonzero()

        f, axarr = plt.subplots(5, 5, figsize=(10, 10))

        for i, (ix, iy) in enumerate(np.ndindex(axarr.shape)):

            indx = incorrect_prediction_idx[i]
            img = np.squeeze(data["test_imgs"][indx, :, :])
            ground_truth = data["classes"][data["test_labels"][indx]]
            pred_label = data["classes"][pred_labels[indx]]

            ax = axarr[ix, iy]
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title(f"P: {pred_label},  T:{ground_truth}")
            ax.imshow(img, cmap="gray")

        plt.savefig(self.output().path)

    def output(self):
        """

        Location of the output image.

        :return: luigi.LocalTarget
        """

        p = Path(self.model_definition)
        p_data = Path(self.input()["dataset"].path)

        return luigi.LocalTarget(
            "{}/{}_{}/test_misclassified_sample.png".format(
                self.modelsfolder, p.stem, p_data.stem
            )
        )


# runs the collection of analysis tasks on the
class RunAnalysisOnTestDataTask(luigi.WrapperTask):
    """
    WrapperTask requiring the tasks GenerateClassificationReportTask, PlotConfusionMatrixTask, and PlotIncorrectTask.

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
        """
        Requires the tasks GenerateClassificationReportTask, PlotConfusionMatrixTask, and PlotIncorrectTask.

        :return: List of luigi.Task subclasses
        """
        return [
            GenerateClassificationReportTask(
                self.imgfolder,
                self.hdffolder,
                self.modelsfolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
                self.sigma,
                self.threshold,
                self.rest_as_other,
                self.whiten,
                self.epsilon,
            ),
            PlotConfusionMatrixTask(
                self.imgfolder,
                self.hdffolder,
                self.modelsfolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
                self.sigma,
                self.threshold,
                self.rest_as_other,
                self.whiten,
                self.epsilon,
            ),
            PlotIncorrectTask(
                self.imgfolder,
                self.hdffolder,
                self.modelsfolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
                self.sigma,
                self.threshold,
                self.rest_as_other,
                self.whiten,
                self.epsilon,
            ),
        ]
