# processing images for input to TensorFlow

import luigi
import os
from tqdm import tqdm
import cv2
import h5py
from deepscribe.luigi.image_processing import AddGaussianNoiseTask
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from typing import List


class SubsampleDatasetTask(luigi.Task):
    """
    Selecting classes from larger dataset
    """

    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    num_augment = luigi.IntParameter()
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths

    def requires(self):
        return AddGaussianNoiseTask(
            self.imgfolder, self.hdffolder, self.target_size, self.num_augment
        )

    def run(self):
        with self.output().temporary_path() as temp_output_path:
            new_archive = h5py.File(temp_output_path)

            original_archive = h5py.File(self.input().path)

            for label in tqdm(original_archive.keys(), desc="Processing labels"):

                if label in self.keep_categories:
                    group = new_archive.require_group(label)
                elif self.rest_as_other:
                    group = new_archive.require_group("OTHER")
                else:
                    continue  # don't do anything

                # copy all elements to the group
                for img_group in tqdm(original_archive[label].keys()):
                    # create new subgroups so augmented images stay in the same place
                    new_img_group = group.create_group(img_group)

                    for img in original_archive[label][img_group].keys():

                        npy_img = original_archive[label][img_group][img].value
                        new_dset = new_img_group.create_dataset(img, data=npy_img)

                        # copy all keys in attributes
                        for key, val in original_archive[label][img_group][
                            img
                        ].attrs.items():
                            new_dset.attrs[key] = val

            new_archive.close()
            original_archive.close()

    def output(self):
        return luigi.LocalTarget(
            "{}/{}_{}_{}_aug_{}{}.h5".format(
                self.hdffolder,
                os.path.basename(self.imgfolder),
                self.target_size,
                "_".join([str(cat) for cat in self.keep_categories]),
                self.num_augment,
                "_OTHER" if self.rest_as_other else "",
            )
        )


class AssignDatasetTask(luigi.Task):
    """
    Assigns data to the train/validation/test sets. Returns .npz file with the different
    categories.
    """

    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    num_augment = luigi.IntParameter(default=0)

    def requires(self):
        return SubsampleDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.num_augment,
        )

    def run(self):

        # TODO : validate fractions

        # loads all data into memory.
        # TODO: not this.

        data_archive = h5py.File(self.input().path)

        images = []
        labels = []

        for label in data_archive.keys():

            for img_group in data_archive[label].keys():

                # keep all images in the same group together

                all_group = [
                    np.array(data_archive[label][img_group][img].value)
                    for img in data_archive[label][img_group].keys()
                ]

                images.append(all_group)
                labels.append(label)

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

        np.savez(
            self.output().path,
            train_imgs=AssignDatasetTask.flatten_and_stack(train_imgs),
            test_imgs=AssignDatasetTask.flatten_and_stack(test_imgs),
            valid_imgs=AssignDatasetTask.flatten_and_stack(valid_imgs),
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
            "{}_split.npz".format(os.path.splitext(self.input().path)[0])
        )
