# processing images for input to TensorFlow

import luigi
import os
from tqdm import tqdm
import cv2
import h5py
from deepscribe.luigi.image_processing import RescaleImageValuesTask
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


class SubsampleDatasetTask(luigi.Task):
    """
    Selecting classes from larger dataset
    """

    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()

    def requires(self):
        return RescaleImageValuesTask(self.imgfolder, self.hdffolder, self.target_size)

    def run(self):
        with self.output().temporary_path() as self.temp_output_path:
            new_archive = h5py.File(self.temp_output_path)

            original_archive = h5py.File(self.input().path)

            for label in tqdm(self.keep_categories, desc="Processing labels"):
                group = new_archive.require_group(label)

                for img in tqdm(original_archive[label].keys()):
                    npy_img = original_archive[label][img].value
                    new_dset = group.create_dataset(img, data=npy_img)
                    # TODO: further abstraction?
                    new_dset.attrs["image_uuid"] = original_archive[label][img].attrs[
                        "image_uuid"
                    ]
                    new_dset.attrs["obj_uuid"] = original_archive[label][img].attrs[
                        "obj_uuid"
                    ]
                    new_dset.attrs["sign"] = original_archive[label][img].attrs["sign"]
                    new_dset.attrs["origin"] = original_archive[label][img].attrs[
                        "origin"
                    ]

            new_archive.close()
            original_archive.close()

    def output(self):
        return luigi.LocalTarget(
            "{}/{}_{}.h5".format(
                self.hdffolder,
                os.path.basename(self.imgfolder),
                "_".join([str(cat) for cat in self.keep_categories]),
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

    def requires(self):
        return SubsampleDatasetTask(
            self.imgfolder, self.hdffolder, self.target_size, self.keep_categories
        )

    def run(self):

        # TODO : validate fractions

        # loads all data into memory.
        # TODO: not this.

        original_archive = h5py.File(self.input().path)

        images = []
        labels = []

        for label in tqdm(original_archive.keys()):

            for img in tqdm(original_archive[label].keys()):
                # creating copy in memory
                npy_img = np.array(original_archive[label][img].value)
                images.append(npy_img)
                labels.append(label)

        original_archive.close()
        # create categorical labels

        enc = LabelEncoder()

        categorical_labels = enc.fit_transform(labels)

        stacked = np.stack(images, axis=0)

        # add extra channel dimension so it doesn't freak out
        #

        stacked = np.expand_dims(stacked, axis=-1)

        # train/test split
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            stacked, categorical_labels, test_size=self.fractions[2]
        )

        # split again for validation

        # compute validation split

        valid_split = self.fractions[1] / (self.fractions[0] + self.fractions[1])

        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            train_imgs, train_labels, test_size=valid_split
        )

        np.savez(
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
            "{}_split.npz".format(os.path.splitext(self.input().path)[0])
        )
