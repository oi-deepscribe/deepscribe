# luigi tasks for image processing
#
import cv2
import luigi
import os
import h5py
from tqdm import tqdm
from deepscribe.pipeline.aggregation import OchreToHD5Task
from sklearn.preprocessing import StandardScaler
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import List


class ProcessImageTask(luigi.Task):
    """
    Task mapping an hdf5 archive of images to another after applying an image processing function.
    """

    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    identifier = ""  # to be set to describe image transformation

    def process_image(self, img):
        raise NotImplementedError

    def run(self):
        with self.output().temporary_path() as self.temp_output_path:
            new_archive = h5py.File(self.temp_output_path)

            original_archive = h5py.File(self.input().path)

            for label in tqdm(original_archive.keys(), desc="Processing labels"):
                group = new_archive.require_group(label)

                for img in original_archive[label].keys():
                    npy_img = original_archive[label][img].value
                    # casting to float32
                    processed_img = self.process_image(npy_img).astype(np.float32)

                    new_dset = group.create_dataset(img, data=processed_img)

                    for key, val in original_archive[label][img].attrs.items():
                        new_dset.attrs[key] = val

            new_archive.close()
            original_archive.close()

    def output(self):

        # append the rest of the parameter values

        additional_params = [
            param
            for param, obj in self.get_params()
            if param not in ["hdffolder", "imgfolder"]
        ]

        additional_param_vals = [
            str(self.__getattribute__(param)) for param in additional_params
        ]

        return luigi.LocalTarget(
            "{}/{}_{}_{}.h5".format(
                self.hdffolder,
                os.path.basename(self.imgfolder),
                self.identifier,
                "_".join(additional_param_vals),
            )
        )


# subtracting out the mean brightness
class RescaleImageValuesTask(ProcessImageTask):
    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    identifier = "rescaled"

    def requires(self):
        return OchreToHD5Task(self.imgfolder, self.hdffolder)

    def process_image(self, img):

        scaled = img / 255.0

        return scaled - np.mean(scaled)


class StandardizeImageSizeTask(ProcessImageTask):
    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    sigma = luigi.FloatParameter(
        default=0.0
    )  # optional, if not using this set it to zero
    identifier = "resized"

    def requires(self):
        return RescaleImageValuesTask(self.imgfolder, self.hdffolder)

    def process_image(self, img):

        """from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#using-opencv.

        Parameters
        ----------
        im : np.array
            Description of parameter `img`.
        target_size : int
            Description of parameter `new_size`.

        Returns
        -------
        np.array
            Description of returned object.

        """

        old_size = img.shape[:2]  # old_size is in (height, width) format
        # computes
        ratio = float(self.target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # new_size should be in (width, height) format
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = int(self.target_size) - new_size[1]
        delta_h = int(self.target_size) - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # filtering here before padding! otherwise, filter will be applied to the border as well.
        img_filtered = gaussian_filter(img, sigma=float(self.sigma))

        new_im = cv2.copyMakeBorder(
            img_filtered, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

        return new_im
