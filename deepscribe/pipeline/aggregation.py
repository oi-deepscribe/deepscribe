# luigi task that iterates through directory of images and saves them all to HDF5 file

import luigi
import os
from tqdm import tqdm
import cv2
import h5py
import unicodedata


class OchreDatasetTask(luigi.ExternalTask):
    """
    An ExternalTask requiring the presence of folder containing images of PFA hotspots.

    """

    imgfolder = luigi.Parameter()

    def output(self):
        """
        Checks if the folder is present.

        :return:
        """

        return luigi.LocalTarget(self.imgfolder)


class OchreToHD5Task(luigi.Task):
    """

    Converts a folder of OCHRE hotspots into a single HDF5 archive

    """

    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()

    def requires(self):
        """
        Requires the presence of the OCHRE Dataset folder

        :return: OchreDatasetTask
        """

        return OchreDatasetTask(self.imgfolder)

    def run(self):

        """

        Copies the images into a single HDF5 archive with metadata from the filenames preserved.

        :return: None
        """

        # open temporary path

        with self.output().temporary_path() as self.temp_output_path:
            archive = h5py.File(self.temp_output_path, "w")
            # iterate through directory, copy images

            for f in tqdm(os.listdir(self.input().path)):

                # normalize filepaths!
                file = unicodedata.normalize("NFC", f)

                # loading image from disk in grayscale
                img = cv2.imread(
                    "{}/{}".format(self.input().path, file), cv2.IMREAD_GRAYSCALE
                )

                if img is not None:
                    # parsing filename according to OCHRE spec
                    fname = os.path.splitext(file)[0]

                    sign, image_uuid, obj_uuid = fname.split("_")
                    # creating groups for signs
                    # requiring bytestrings
                    current_group = archive.require_group(sign)

                    dset = current_group.create_dataset(fname, data=img)
                    dset.attrs["image_uuid"] = image_uuid
                    dset.attrs["obj_uuid"] = obj_uuid
                    dset.attrs["sign"] = sign
                    dset.attrs["origin"] = "{}/{}".format(self.input().path, file)

            archive.close()

    def output(self):
        """

        HDF5 archive containing contents of the folder.

        :return: luigi.LocalTarget
        """

        return luigi.LocalTarget(
            "{}/{}.h5".format(self.hdffolder, os.path.basename(self.imgfolder))
        )
