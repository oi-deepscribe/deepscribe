import matplotlib.pyplot as plt
import luigi
from .training import TrainKerasModelTask
from .selection import SelectDatasetTask
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from scipy.stats import entropy
from luigi.util import requires
from abc import ABC
import os
import tensorflow.keras as kr
import tensorflow as tf
import matplotlib
from pathlib import Path
import json

matplotlib.use("Agg")


@requires(TrainKerasModelTask, SelectDatasetTask)
class PlotConfusionMatrixTask(luigi.Task):
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
        model = kr.models.load_model(self.input()[0].path)
        data = np.load(self.input()[1].path)

        # make predictions on data

        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])

        # computing predicted labels
        pred_labels = np.argmax(pred_logits, axis=1)

        # compute confusion matrix

        confusion = confusion_matrix(data["test_labels"], pred_labels)

        # get labels from dataset

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

        # assumes output is a single file

        model_parent = Path(self.input()[0].path).parents[0]

        return luigi.LocalTarget(f"{model_parent}/confusion_test.png")


@requires(TrainKerasModelTask, SelectDatasetTask)
class GenerateClassificationReportTask(luigi.Task):
    """

    Generates a classification report from the trained model on test data. 

    """

    def run(self):
        """
        Loads the model, runs it on test data, saves the classification report as txt.


        :return: None
        """
        # load TF model and dataset
        model = kr.models.load_model(self.input()[0].path)
        data = np.load(self.input()[1].path)

        # make predictions on data

        # TEST DATA

        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])

        # computing predicted labels
        pred_labels = np.argmax(pred_logits, axis=1)

        # print(data["test_labels"].shape)
        # print(pred_labels.shape)

        report = classification_report(
            data["test_labels"], pred_labels, target_names=data["classes"]
        )

        # Compute the top-k accuracy across all of the data:

        top_k_test = []

        for i in range(2, min(6, len(data["classes"]))):
            k_i = kr.metrics.sparse_top_k_categorical_accuracy(
                data["test_labels"], pred_logits, k=i
            )

            top_k_test.append(f"top-{i} accuracy: {np.mean(k_i)}")

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
        for i in range(2, min(6, len(data["classes"]))):
            k_i = kr.metrics.sparse_top_k_categorical_accuracy(
                data["train_labels"], pred_logits, k=i
            )

            top_k_train.append(f"top-{i} accuracy: {np.mean(k_i)}")

        with self.output().temporary_path() as temppath:
            with open(temppath, "w") as outf:
                outf.write("REPORT ON TEST\n")
                outf.write(report)
                outf.write("TOP-K ACCURACIES ON TEST\n")
                outf.write("\n".join(top_k_test))
                outf.write("\n")
                outf.write("REPORT ON TRAIN\n")
                outf.write(report_train)
                outf.write("TOP-K ACCURACIES ON TRAIN\n")
                outf.write("\n".join(top_k_train))

    def output(self):

        # assumes output is a single file

        model_parent = Path(self.input()[0].path).parents[0]

        return luigi.LocalTarget(f"{model_parent}/classification_report.png")


# plots a random sample of 16 incorrect images from test.
@requires(TrainKerasModelTask, SelectDatasetTask)
class PlotIncorrectTest(luigi.Task):
    """

    Loads 16 incorrectly classified images from test data and plots them in a grid.

    """

    outname = "test_misclassified_sample.png"

    def run(self):
        """
        Loads model, runs on test data, picks 16 random incorrectly classified images.

        :return:
        """

        # load TF model and dataset
        model = kr.models.load_model(self.input()[0].path)
        data = np.load(self.input()[1].path)
        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])
        # (batch_size,)

        pred_labels = np.argmax(pred_logits, axis=1)

        (incorrect_prediction_idx,) = np.not_equal(
            data["test_labels"], pred_labels
        ).nonzero()

        _, axarr = plt.subplots(5, 5, figsize=(10, 10))

        for i, (ix, iy) in enumerate(np.ndindex(axarr.shape)):

            indx = incorrect_prediction_idx[i]
            img = np.squeeze(data["test_imgs"][indx, :, :])
            ground_truth = data["classes"][data["test_labels"][indx]]
            pred_label = data["classes"][pred_labels[indx]]
            h = entropy(pred_logits[indx, :])

            normalized = h / np.log(len(data["classes"]))

            ax = axarr[ix, iy]
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title(f"P:{pred_label},T:{ground_truth},H:{round(normalized, 2)}")
            ax.imshow(img, cmap="gray")

        plt.savefig(self.output().path)

    def output(self):

        # assumes output is a single file

        model_parent = Path(self.input()[0].path).parents[0]

        return luigi.LocalTarget(f"{model_parent}/test_misclassified_sample.png")


# plots a random sample of 16 incorrect images from test.
@requires(TrainKerasModelTask, SelectDatasetTask)
class PlotIncorrectTrain(luigi.Task):
    """

    Loads 16 incorrectly classified images from test data and plots them in a grid.

    """

    outname = "test_misclassified_sample.png"

    def run(self):
        """
        Loads model, runs on test data, picks 16 random incorrectly classified images.

        :return:
        """

        # load TF model and dataset
        model = kr.models.load_model(self.input()[0].path)
        data = np.load(self.input()[1].path)
        # (batch_size, num_classes)
        pred_logits = model.predict(data["train_imgs"])
        # (batch_size,)

        pred_labels = np.argmax(pred_logits, axis=1)

        (incorrect_prediction_idx,) = np.not_equal(
            data["train_labels"], pred_labels
        ).nonzero()

        _, axarr = plt.subplots(5, 5, figsize=(10, 10))

        for i, (ix, iy) in enumerate(np.ndindex(axarr.shape)):

            indx = incorrect_prediction_idx[i]
            img = np.squeeze(data["train_imgs"][indx, :, :])
            ground_truth = data["classes"][data["train_labels"][indx]]
            pred_label = data["classes"][pred_labels[indx]]
            h = entropy(pred_logits[indx, :])

            normalized = h / np.log(len(data["classes"]))

            ax = axarr[ix, iy]
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title(f"P:{pred_label},T:{ground_truth},H:{round(normalized, 2)}")
            ax.imshow(img, cmap="gray")

        plt.savefig(self.output().path)

    def output(self):

        # assumes output is a single file

        model_parent = Path(self.input()[0].path).parents[0]

        return luigi.LocalTarget(f"{model_parent}/train_misclassified_sample.png")


# runs the collection of analysis tasks on the
@requires(
    PlotConfusionMatrixTask,
    GenerateClassificationReportTask,
    PlotIncorrectTrain,
    PlotIncorrectTest,
)
class TrainAndAnalyze(luigi.WrapperTask):
    """
    WrapperTask requiring the tasks GenerateClassificationReportTask, 
    PlotConfusionMatrixTask, 
    and PlotIncorrectTask.

    """
