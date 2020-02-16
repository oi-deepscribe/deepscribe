# processing images for input to TensorFlow

import luigi
import os
from tqdm import tqdm
import h5py
from deepscribe.pipeline.images import GaussianBlurTask
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from typing import List


class SelectDatasetTask(luigi.Task):
    """
    Selecting classes from larger dataset.
    Assigns data to the train/validation/test sets.
    Returns .npz file with the different
    categories.
    """

    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    sigma = luigi.FloatParameter(default=0.5)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return GaussianBlurTask(
            self.imgfolder, self.hdffolder, self.target_size, self.sigma
        )

    def run(self):

        if sum(self.fractions) != 1.0:
            raise ValueError(
                f"Invalid split {self.fractions} passed to --fractions argument. Fractions should add up to 1."
            )

        # loads all data into memory.
        # TODO: not this. Maybe instead of creating the NPY archive we could stream from the HDF archive?
        # Convert to TF Data workflow?

        data_archive = h5py.File(self.input().path)

        # TODO: find total array dimensions from hdf archive

        images_lst = []
        labels_lst = []

        original_archive = h5py.File(self.input().path)

        for label in tqdm(original_archive.keys(), desc="Selecting labels"):

            if label in self.keep_categories or self.rest_as_other:

                all_label_imgs = [
                    np.array(data_archive[label][img].value)
                    for img in data_archive[label].keys()
                ]

                images_lst.extend(all_label_imgs)

                if label in self.keep_categories:
                    labels_lst.extend([label for img in all_label_imgs])
                elif self.rest_as_other:
                    labels_lst.extend(["OTHER" for img in all_label_imgs])

        data_archive.close()
        # create categorical labels
        enc = LabelEncoder()

        fracs = np.array(self.fractions)

        categorical_labels = enc.fit_transform(labels_lst)

        # [n_images, img_dim, img_dim, n_channels] (n_channels) is 1 in this case
        images = np.expand_dims(np.stack(images_lst, axis=0), axis=-1)

        # train/test split
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            images, categorical_labels, test_size=fracs[2]
        )

        # split again for validation

        # compute validation split

        valid_split = fracs[1] / (fracs[0] + fracs[1])

        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            train_imgs, train_labels, test_size=valid_split
        )

        # flatten arrays and expand dims

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
        return luigi.LocalTarget(
            "{}/{}_{}_{}_{}{}.npz".format(
                self.hdffolder,
                os.path.basename(self.imgfolder),
                self.target_size,
                "_".join([str(cat) for cat in self.keep_categories]),
                self.sigma,
                "_OTHER" if self.rest_as_other else "",
            )
        )
