# processing images for input to TensorFlow

import luigi
import os
from tqdm import tqdm
import h5py
from deepscribe.pipeline.images import AddGaussianNoiseTask
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
    num_augment = luigi.IntParameter(default=0)
    sigma = luigi.FloatParameter(default=0.5)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return AddGaussianNoiseTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.sigma,
            self.num_augment,
        )

    def run(self):

        if sum(self.fractions) != 1.0:
            raise ValueError(
                f"Invalid split {self.fractions} passed to --fractions argument. Fractions should add up to 1."
            )

        # loads all data into memory.
        # TODO: not this. Maybe instead of creating the NPY archive we should stream from the HDF archive?
        # Convert to TF Data workflow?

        data_archive = h5py.File(self.input().path)

        images = []
        labels = []

        original_archive = h5py.File(self.input().path)

        for label in tqdm(original_archive.keys(), desc="Selecting labels"):

            for img_group in data_archive[label].keys():

                if label in self.keep_categories or self.rest_as_other:

                    all_group_imgs = [
                        np.array(data_archive[label][img_group][img].value)
                        for img in data_archive[label][img_group].keys()
                    ]
                    # appending as a group to keep augmented images of the same class together.
                    images.append(all_group_imgs)

                    if label in self.keep_categories:
                        labels.append(label)
                    elif self.rest_as_other:
                        images.append("OTHER")

        data_archive.close()
        # create categorical labels
        enc = LabelEncoder()

        categorical_labels = enc.fit_transform(labels)

        # train/test split
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            images, categorical_labels, test_size=self.fractions[2]
        )

        # split again for validation

        # compute validation split

        valid_split = self.fractions[1] / (self.fractions[0] + self.fractions[1])

        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            train_imgs, train_labels, test_size=valid_split
        )

        # flatten arrays and expand dims

        # repeat train and test labels

        np.savez_compressed(
            self.output().path,
            train_imgs=SelectDatasetTask.flatten_and_stack(train_imgs),
            test_imgs=SelectDatasetTask.flatten_and_stack(test_imgs),
            valid_imgs=SelectDatasetTask.flatten_and_stack(valid_imgs),
            train_labels=np.tile(train_labels, self.num_augment + 1),
            test_labels=np.tile(test_labels, self.num_augment + 1),
            valid_labels=np.tile(valid_labels, self.num_augment + 1),
            classes=enc.classes_,
        )

    # flattens a list of lists of numpy arrays and stacks them ino one array with
    # an expanded dimension. Used to keep labels of augmented images consistent.
    @staticmethod
    def flatten_and_stack(lst: List[List[np.array]]) -> np.array:

        flattened_list = list(itertools.chain.from_iterable(lst))
        stacked = np.stack(flattened_list, axis=0)
        stacked = np.expand_dims(stacked, axis=-1)

        return stacked

    def output(self):
        return luigi.LocalTarget(
            "{}/{}_{}_{}_{}_aug_{}{}.npz".format(
                self.hdffolder,
                os.path.basename(self.imgfolder),
                self.target_size,
                "_".join([str(cat) for cat in self.keep_categories]),
                self.sigma,
                self.num_augment,
                "_OTHER" if self.rest_as_other else "",
            )
        )
