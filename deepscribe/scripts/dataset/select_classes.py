# Selecting classes from the dataset to build toy examples.

from argparse import ArgumentParser
from sys import argv
import csv
import cv2
from tqdm import tqdm
import os

def parse(args):
    """
    Parse stuff.
    """
    parser = ArgumentParser()
    parser.add_argument("--datafiles", nargs='*', help="CSV files with columns 'Name' and 'Primary Image'")
    parser.add_argument("--imgfolder", help="folder with training images.")
    parser.add_argument("--remove_prefix", required=False, help="prefix to remove, if desired.")
    parser.add_argument("--classes", nargs="*", help="classes to select out")
    parser.add_argument("--out", help="output folder for data")
    return parser.parse_args(args)

def run(args):
    """
    Do stuff.
    """

    all_data = []
    for filename in args.datafiles:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            #VALIDATING INPUT DATA
            all_data.extend([(row['Name'], row['Primary Image']) for row in reader if os.path.splitext(row['Primary Image'])[1]==".jpg"])

    print("{} data points loaded".format(len(all_data)))

    # removing target prefix from dataset
    if args.remove_prefix:
        all_data = [(name, img_file) for name, img_file in all_data if not name.startswith(args.remove_prefix)]

    print("{} data points remaining".format(len(all_data)))


    # building dictionary with key: class name, value: list of examples
    class_examples = {}
    original_filenames = {}
    for class_name in args.classes:
        # select image files
        img_files = [img_file for name, img_file in all_data if name == class_name]
        # load image files from disk
        original_filenames[class_name] = img_files
        class_examples[class_name] = [cv2.imread("{}/{}".format(args.imgfolder, img_file), 0) for img_file in tqdm(img_files)]
        print("class {} has {} examples".format(class_name, len(class_examples[class_name])))

    #save images to disk
    #
    for class_name in args.classes:
        outdir = "{}/{}".format(args.out, class_name)
        try:
            os.mkdir(outdir)
        except FileExistsError as e:
            pass

        for i, img in enumerate(tqdm(class_examples[class_name])):
            cv2.imwrite("{}/{}".format(outdir, original_filenames[class_name][i]), img)

    print("saved dataset to disk")



if __name__=="__main__":
    run(parse(argv[1:]))
