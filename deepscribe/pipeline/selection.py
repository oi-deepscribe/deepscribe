# processing images for input to TensorFlow

import luigi
import os
from tqdm import tqdm
import h5py
from deepscribe.pipeline.images import StandardizeImageSizeTask
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    threshold = luigi.BoolParameter(default=False)
    rest_as_other = luigi.BoolParameter(
        default=False
    )  # set the remaining as "other" - not recommended for small keep_category lengths
    whiten = luigi.BoolParameter(
        default=False
    )  # perform ZCA whitening on whole dataset
    epsilon = luigi.FloatParameter(default=0.1)

    def requires(self):
        return StandardizeImageSizeTask(
            self.imgfolder, self.hdffolder, self.target_size, self.sigma, self.threshold
        )

    def run(self):

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

        for label in tqdm(self.keep_categories, desc="Selecting Labels"):
            all_label_imgs = [
                np.array(data_archive[label][img]) for img in data_archive[label].keys()
            ]

            images_lst.extend(all_label_imgs)

            labels_lst.extend([label for img in all_label_imgs])

        # getting the rest. Slight code reuse but easier to debug.
        if self.rest_as_other:
            other_labels = [
                label
                for label in data_archive.keys()
                if label not in self.keep_categories
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
            images, categorical_labels, test_size=fracs[2]
        )

        # split again for validation

        valid_split = fracs[1] / (fracs[0] + fracs[1])

        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            train_imgs, train_labels, test_size=valid_split
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
        return luigi.LocalTarget(
            "{}/{}_{}_{}_{}{}{}{}.npz".format(
                self.hdffolder,
                os.path.basename(self.imgfolder),
                self.target_size,
                "_".join([str(cat) for cat in self.keep_categories]),
                self.sigma,
                "_OTHER" if self.rest_as_other else "",
                f"_whitened_{self.epsilon}" if self.whiten else "",
                "threshed" if self.threshold else "",
            )
        )
