# luigi tasks for image processing
#
import cv2
import luigi
import os
import h5py
from tqdm import tqdm
from deepscribe.luigi.preprocessing import OchreToHD5Task
from sklearn.preprocessing import StandardScaler


class ProcessImageTask(luigi.Task):
    '''
    Task mapping an hdf5 archive of images to another after applying an image processing function.
    '''
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    identifier = "" # to be set to describe image transformation


    def process_image(self, img):
        raise NotImplementedError

    def run(self):
        with self.output().temporary_path() as self.temp_output_path:
            new_archive = h5py.File(self.temp_output_path)

            original_archive = h5py.File(self.input().path)

            for label in tqdm(original_archive.keys(), desc="Processing labels"):
                group = new_archive.require_group(label)

                for img in tqdm(original_archive[label].keys()):
                    npy_img = original_archive[label][img].value
                    processed_img = self.process_image(npy_img)
                    new_dset = group.create_dataset(img, data=processed_img)
                    #TODO: further abstraction?
                    new_dset.attrs['image_uuid'] = original_archive[label][img].attrs['image_uuid']
                    new_dset.attrs['obj_uuid'] = original_archive[label][img].attrs['obj_uuid']
                    new_dset.attrs['sign'] = original_archive[label][img].attrs['sign']
                    new_dset.attrs['origin'] = original_archive[label][img].attrs['origin']

            new_archive.close()
            original_archive.close()


    def output(self):
        return luigi.LocalTarget("{}/{}_{}.h5".format(self.hdffolder,
                                                    os.path.basename(self.imgfolder),
                                                    self.identifier))

class ImagesToGrayscaleTask(ProcessImageTask):
    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    identifier = "grayscale"

    def requires(self):
        return OchreToHD5Task(self.imgfolder, self.hdffolder)

    def process_image(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#TODO: do this with only padding or only resizing/scaling?
class StandardizeImageSizeTask(ProcessImageTask):
    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter() #standardizing to square images
    identifier = "standardized"

    def requires(self):
        return ImagesToGrayscaleTask(self.imgfolder, self.hdffolder)

    def process_image(self, img):

        # TODO: update
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

        old_size = img.shape[:2] # old_size is in (height, width) format
        # computes
        ratio = float(self.target_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        im = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = self.target_size - new_size[1]
        delta_h = self.target_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        return new_im

class RescaleImageValuesTask(ProcessImageTask):
    # location of image folder
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    target_size = luigi.IntParameter() #standardizing to square images
    identifier = "rescaled"

    def requires(self):
        return StandardizeImageSizeTask(self.imgfolder, self.hdffolder, self.target_size)

    def process_image(self, img):
        return StandardScaler().fit_transform(img)
