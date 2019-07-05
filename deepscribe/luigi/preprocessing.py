# luigi task that iterates through directory of images and saves them all to HDF5 file

import luigi
import os
from tqdm import tqdm
import cv2
import h5py

class OchreDatasetTask(luigi.ExternalTask):
    imgfolder = luigi.Parameter()
    def output(self):
        return luigi.LocalTarget(self.imgfolder)

class OchreToHD5Task(luigi.Task):
    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()

    def requires(self):
        return OchreDatasetTask(self.imgfolder)

    def run(self):

        # open temporary path

        with self.output().temporary_path() as self.temp_output_path:
            archive = h5py.File(self.temp_output_path)
            # iterate through directory, copy images


            for file in tqdm(os.listdir(self.input().path)):
                #loading image from disk
                img = cv2.imread("{}/{}".format(self.input().path, file))

                if img is not None:
                    #parsing filename according to OCHRE spec
                    fname = os.path.splitext(file)[0]

                    sign, image_uuid, obj_uuid = fname.split("_")
                    # creating groups for signs
                    current_group = archive.require_group(sign)

                    dset = current_group.create_dataset(fname, data=img)
                    dset.attrs['image_uuid'] = image_uuid
                    dset.attrs['obj_uuid'] = obj_uuid
                    dset.attrs['sign'] = sign
                    dset.attrs['origin'] = "{}/{}".format(self.input().path, file)

            archive.close()

    def output(self):
        return luigi.LocalTarget("{}/{}.h5".format(self.hdffolder,
                                                    os.path.basename(self.imgfolder)))
