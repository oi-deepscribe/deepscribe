# luigi tasks for image processing
#
import cv2
import luigi
import os
import h5py
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from skimage.util import random_noise
from abc import ABC
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import List


class ProcessImagesTask(luigi.Task):
    """
    Task mapping an hdf5 archive of images to another after applying an image processing function.
    

    """

    imgarchive = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    histogram = luigi.Parameter(default="")  # adaptive,
    sigma = luigi.FloatParameter(default=0.0)  # gaussian blur parameter

    def process_image(self, img):
        """

        Task mapping an image to to a transformed image.

        padding from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#using-opencv.

        :param img: np.ndarray
        :return: np.ndarray
        """

        # resized image
        old_size = img.shape[:2]  # old_size is in (height, width) format
        # computes
        ratio = float(self.target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # new_size should be in (width, height) format
        img = cv2.resize(img, (new_size[1], new_size[0]))

        # optional gaussian blur.
        img = gaussian_filter(img, sigma=float(self.sigma))

        if self.histogram == "adaptive":
            # compute CLAHE adaptive histogram
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        elif self.histogram == "equalized":
            img = cv2.equalizeHist(img)

        # standardize
        img = (img - np.mean(img)) / np.std(img)

        # filtering here before padding! otherwise, filter will be applied to the border as well.

        delta_w = int(self.target_size) - new_size[1]
        delta_h = int(self.target_size) - new_size[0]

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

        return img

    def run(self):
        """

        Runs the process_image function on every image in the HDF5 archive, keeping the archive structure constant.

        :return:
        """

        with self.output().temporary_path() as self.temp_output_path:
            new_archive = h5py.File(self.temp_output_path, "w")

            original_archive = h5py.File(self.imgarchive, "r")

            for label in tqdm(original_archive.keys(), desc="Processing labels"):
                group = new_archive.require_group(label)

                for img in original_archive[label].keys():
                    npy_img = np.array(original_archive[label][img])
                    # casting to float32
                    processed_img = self.process_image(npy_img).astype(np.float32)

                    new_dset = group.create_dataset(img, data=processed_img)

                    for key, val in original_archive[label][img].attrs.items():
                        new_dset.attrs[key] = val

            new_archive.close()
            original_archive.close()

    def output(self):
        """

        The location of the transformed HDF5 archive on disk.

        :return: luigi.LocalTarget
        """

        pathroot = Path(self.imgarchive).with_suffix("")

        return luigi.LocalTarget(
            "{}_processed_{}{}{}_{}.h5".format(
                pathroot,
                "sigma",
                str(self.sigma).replace(".", "p"),
                self.histogram,
                self.target_size,
            )
        )
