# Builds a .npz dataset from a dataset of images:

#for a folder structured like root/{label}/{imgs},
# collects data and produces a .npz archive with stacked images and labels

from argparse import ArgumentParser
from sys import argv
import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


def parse(args):
    """
    Parse stuff.
    """
    parser = ArgumentParser()
    parser.add_argument("--dataset", help="folder with dataset")
    parser.add_argument("--npz", help="location of output NPZ file")
    return parser.parse_args(args)

def run(args):
    """
    Do stuff.
    """

    print("Extracting data from {}".format(args.dataset))

    labels = [name for name in os.listdir(args.dataset) if os.path.isdir("{}/{}".format(args.dataset,name))]

    dataset = [] #list of lists of numpy arrays
    for label in labels:
        img_names = glob.glob("{}/{}/*.jpg".format(args.dataset, label))
        # loading images as grayscale
        #
        imgs = [cv2.imread(img_name, 0) for img_name in img_names]
        # creating corresponding label array
        label_array = np.array([label] * len(imgs))

        dataset.append((np.stack(imgs), label_array, np.array(img_names))) # producing numpy array



    # stacking all together
    #
    imgs, labels, img_names = zip(*dataset)

    all_imgs = np.vstack(imgs)
    # adding extra dimension b/c grayscale
    all_imgs_reshaped = np.reshape(all_imgs, all_imgs.shape + (1,))
    all_labels = np.concatenate(labels)
    all_names = np.concatenate(img_names)

    # encode labels as digits
    le = LabelEncoder()
    labels_encoded = le.fit_transform(all_labels)

    print("saving to {}".format(args.npz))

    np.savez_compressed(args.npz, imgs=all_imgs_reshaped, labels=labels_encoded, original_labels=all_labels, names=all_names)



if __name__=="__main__":
    run(parse(argv[1:]))
