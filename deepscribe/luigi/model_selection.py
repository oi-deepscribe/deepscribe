import luigi
from .training import TrainKerasModelFromDefinitionTask
from .ml_input import AssignDatasetTask
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import tensorflow.keras as kr
import matplotlib
from pylatex import Document, Section, Figure, NoEscape

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

        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])

        # computing predicted labels
        pred_labels = np.argmax(pred_logits, axis=1)

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
    num_augment = luigi.IntParameter(default=0)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return {
            "confusion_matrix": TestModelTask(
                self.imgfolder,
                self.hdffolder,
                self.modelsfolder,
                self.target_size,
                self.keep_categories,
                self.fractions,
                self.model_definition,
                self.num_augment,
                self.rest_as_other
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
        # load matrix

        confusion = np.load(self.input()["confusion_matrix"].path)

        # get labels from dataset

        data = np.load(self.input()["dataset"].path)

        class_labels = data["classes"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("Confusion matrix from \n {}".format(self.input()["dataset"].path))
        cax = ax.matshow(confusion)
        fig.colorbar(cax)


        # appending extra tick label to make sure everything aligns properly
        ax.set_xticklabels(list(class_labels[:1]) + list(class_labels))
        ax.set_yticklabels(list(class_labels[:1]) + list(class_labels))
        plt.savefig(self.output().path)

    def output(self):
        return luigi.LocalTarget(
            "{}_confusion.png".format(os.path.splitext(self.input()["dataset"].path)[0])
        )


class PlotMisclassificationTopKTask(luigi.Task):
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
    k = luigi.IntParameter(default=5)

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

        # (batch_size, num_classes)
        pred_logits = model.predict(data["test_imgs"])

        # (batch_size,)
        in_top_k = kr.backend.in_top_k(pred_logits, data["test_labels"], self.k)

        # get indices of datapoints that aren't in the top k

        num_incorrect = int(np.sum(np.logical_not(in_top_k)))

        # (num_incorrect, img_size_x, img_size_y, img_depth)
        incorrect_top_5 = data["test_imgs"][np.logical_not(in_top_k), :, :, :]
        # (num_incorrect,)
        incorrect_top_5_truth = data["test_labels"][np.logical_not(in_top_k)]
        # (num_incorrect, num_classes)
        incorrect_logits = pred_logits[np.logical_not(in_top_k), :]

        with self.output().temporary_path() as temppath:

            doc = Document(temppath, geometry_options={"right": "2cm", "left": "2cm"})

            doc.append("Introduction.")

            # TODO: more experiment data

            # assemble image, ground truth label, top-5 predicted classes
            for i in range(num_incorrect):
                img = incorrect_top_5[i, :, :, :]
                ground_truth = incorrect_top_5_truth[:]
                top_k_predictions = np.argsort(incorrect_logits[i, :])[0 : self.k]
                top_k_predicted_labels = data["classes"][top_k_predictions]

                with doc.create(Section(f"Misclassified Image {i}")):
                    doc.append(f"Predicted: {top_k_predicted_labels}")
                    fig = plt.figure()
                    plt.imshow(img)
                    with doc.create(Figure(position="htbp")) as plot:
                        plot.add_plot()
                        plot.add_caption(
                            f"Correct Label:{data['classes'][ground_truth]}"
                        )

                    doc.append("Created using matplotlib.")

                    fig.close()

            doc.append("Conclusion.")

            doc.generate_pdf(clean_tex=False)

    def output(self):
        return luigi.LocalTarget(
            "{}_test_misclassified.pdf".format(os.path.splitext(self.input()["dataset"].path)[0])
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
