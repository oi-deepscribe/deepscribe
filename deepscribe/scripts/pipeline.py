#data preprocessing pipeline for OCHRE data.

from argparse import ArgumentParser
from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import csv
import h5py
from PIL import Image
import numpy as np
import os
import errno
from tqdm import tqdm

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--datafiles", nargs='*', help="CSV files with columns 'Name' and 'Primary Image'")
    parser.add_argument("--imgfolder", help="folder with training images.")
    parser.add_argument("--remove_prefix", required=False, help="prefix to remove, if desired.")
    parser.add_argument("--min_size", nargs=2, type=int, required=False, help="minimum size to be kept in each image.")
    parser.add_argument("--examples_req", type=int, help="number of training examples per class required. Classes with too few examples will be skipped.", required=False)
    parser.add_argument("--blur_thresh", type=float, help="Laplacian variance threshold.", required=False)
    parser.add_argument("--edge_detect", action="store_true", help="perform edge detection. ")
    parser.add_argument("--resize", nargs=2, help="output image size, in px.", type=int, required=False)
    parser.add_argument("--outfile", help="location of output HDF5 file. ")
    return parser.parse_args(args)

def run(args):
    # opening all CSV files
    all_data = []
    for filename in args.datafiles:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            all_data.extend([(row['Name'], row['Primary Image']) for row in reader])

    print("{} data points loaded".format(len(all_data)))

    # removing target prefix from dataset
    if args.remove_prefix:
        all_data = [(name, img_file) for name, img_file in all_data if not name.startswith(args.remove_prefix)]

    print("{} data points remaining".format(len(all_data)))
    # filtering out by number of examples present
    if args.examples_req:
        names , _ = zip(*all_data)
        unique, counts = np.unique(names, return_counts=True)
        valid_names = unique[counts >= args.examples_req]
        all_data = [(name, img_file) for name, img_file in all_data if name in valid_names]

    print("{} data points remaining".format(len(all_data)))
    # load all images to check and filter for image-specific properties

    # loading as grayscale
    img_data = [(cv2.imread("{}/{}".format(args.imgfolder, img_file), 0), name) for name, img_file in all_data]


    # filter out images that didn't load due to database error

    img_data = [(img, label) for img, label in img_data if isinstance(img, np.ndarray)]

    if args.min_size:
        min_size = np.array(args.min_size)
        # if both dimensions are larger than the minimum size
        img_data = [(img, label) for img, label in img_data if np.all(img.shape > min_size)]

    print("{} data points remaining".format(len(img_data)))

    if args.blur_thresh:
        # filter out using variance of laplacian (see https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
        img_data = [(img, label) for img, label in img_data if cv2.Laplacian(img, cv2.CV_64F).var() > args.blur_thresh]

    print("{} data points remaining".format(len(img_data)))

    # if args.edge_detect:
    #     img_data = [(img, label) for img, label in img_data]


    # resize images
    if args.resize:
        img_data = [(cv2.resize(img, tuple(args.resize)), label) for img, label in img_data]


    # stack all images
    imgs, labels = zip(*img_data)

    labels_stacked = np.array(labels)

    # encode labels as digits
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_stacked)

    imgs = [np.array(im) for im in imgs]

    imgs_stacked = np.array(imgs)
    # adding extra dimension b/c grayscale
    imgs_stacked = np.reshape(imgs, imgs_stacked.shape + (1,))

    np.savez_compressed(args.outfile, labels=labels_encoded, decoded=labels_stacked, imgs=imgs_stacked)

if __name__=="__main__":
    run(parse(argv[1:]))
