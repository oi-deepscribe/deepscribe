# resizing and padding images

from argparse import ArgumentParser
from sys import argv
from deepscribe.imageproc.padding import resize_pad
from os import listdir, mkdir, path
import cv2
from tqdm import tqdm
import glob
import os

def parse(args):
    """
    Parse stuff.
    """
    parser = ArgumentParser()
    parser.add_argument("--infolder", help="folder of input images.")
    parser.add_argument("--outfolder", help="folder of output images.")
    return parser.parse_args(args)

def run(args):
    """
    Loads all images, performs padding operation, saves them to disk.
    """

    try:
        mkdir(args.outfolder)
    except FileExistsError as e:
        pass


    filenames = glob.glob("{}/*.jpg".format(args.infolder))

    imgs = [cv2.imread(img) for img in filenames]

    # gaussian blur
    #
    processed_images = [cv2.GaussianBlur(img,(5,5),0) for img in imgs]


    for img, filename in zip(processed_images, filenames):
        basename = os.path.basename(filename)
        cv2.imwrite("{}/{}".format(args.outfolder, basename), img)



if __name__=="__main__":
    run(parse(argv[1:]))
