# processing images for input to TensorFlow

import luigi
import os
from tqdm import tqdm
import h5py
from luigi.util import requires
from .images import ProcessImagesTask
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


@requires(ProcessImagesTask)
class SelectDatasetTask(luigi.Task):
    """
    Selecting classes from larger dataset to produce a smaller archive for training.
    Assigns data to the train/validation/test sets.
    Returns .npz file with the different
    categories.
    """

    keep_categories = luigi.Parameter()  # if none, keep all
    fractions = luigi.ListParameter()  # train/valid/test fraction
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths
    whiten = luigi.BoolParameter(
        default=False
    )  # perform ZCA whitening on whole dataset
    epsilon = luigi.FloatParameter(default=0.1)  # ZCA whitening epsilon

    def run(self):
        """

        Splits data into train, test, eval sets and saves to npz archive.

        :return:
        """

        if sum(self.fractions) != 1.0:
            raise ValueError(
                f"Invalid split {self.fractions} passed to --fractions argument. Fractions should add up to 1."
            )

        # loads all data into memory.
        # TODO: investigate streaming directly from HDF5 dataset

        data_archive = h5py.File(self.input().path, "r")

        # TODO: find total array dimensions from hdf archive

        images_lst = []
        labels_lst = []

        # load keep categories from file if present

        if self.keep_categories is not None:
            with open(self.keep_categories, "r") as infile:
                categories = [line.strip() for line in infile.readlines()]
        else:
            categories = list(np.unique(data_archive.keys()))

        for label in tqdm(categories, desc="Selecting Labels"):
            all_label_imgs = [
                np.array(data_archive[label][img]) for img in data_archive[label].keys()
            ]

            images_lst.extend(all_label_imgs)

            labels_lst.extend([label for img in all_label_imgs])

        # getting the rest. Slight code reuse but easier to debug.
        if self.rest_as_other:
            other_labels = [
                label for label in data_archive.keys() if label not in categories
            ]

            for label in tqdm(other_labels, desc="Getting OTHER labels"):
                all_label_imgs = [
                    np.array(data_archive[label][img])
                    for img in data_archive[label].keys()
                ]

                images_lst.extend(all_label_imgs)

                labels_lst.extend(["OTHER" for img in all_label_imgs])

        data_archive.close()
        # create categorical labels
        enc = LabelEncoder()
        categorical_labels = enc.fit_transform(labels_lst)

        # [n_images, img_dim, img_dim, n_channels] (n_channels) is 1 in this case
        images = np.expand_dims(np.stack(images_lst, axis=0), axis=-1)

        # perform ZCA whitening on entire dataset if specified
        if self.whiten:
            datagen = ImageDataGenerator(zca_whitening=True, zca_epsilon=self.epsilon)
            datagen.fit(images)
            # get everything back out

            images, categorical_labels = datagen.flow(
                images, y=categorical_labels, batch_size=images.shape[0]
            ).next()

        fracs = np.array(self.fractions)

        # train/test split
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            images, categorical_labels, test_size=fracs[2], stratify=categorical_labels
        )

        # split again for validation

        valid_split = fracs[1] / (fracs[0] + fracs[1])

        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            train_imgs, train_labels, test_size=valid_split, stratify=train_labels
        )

        # repeat train and test labels

        np.savez_compressed(
            self.output().path,
            train_imgs=train_imgs,
            test_imgs=test_imgs,
            valid_imgs=valid_imgs,
            train_labels=train_labels,
            test_labels=test_labels,
            valid_labels=valid_labels,
            classes=enc.classes_,
        )

    def output(self):
        """

        LocalTarget with path of the output NPZ archive.

        :return: luigi.LocalTarget
        """

        return luigi.LocalTarget(
            "{}_{}{}{}.npz".format(
                Path(self.input().path).resolve().with_suffix(""),
                Path(self.keep_categories).stem if self.keep_categories else "all",
                "_OTHER" if self.rest_as_other else "",
                f"_whitened_{str(self.epsilon).replace('.', 'p')}"
                if self.whiten
                else "",
            )
        )
