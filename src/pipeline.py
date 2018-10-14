#data preprocessing pipeline for OCHRE data.

from argparse import ArgumentParser
from sys import argv
from sklearn.model_selection import train_test_split
import csv
from PIL import Image
import numpy as np
import os
import errno

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--datacsv", help="CSV with columns 'Name' and 'Primary Image'")
    parser.add_argument("--remove_prefix", required=False, help="prefix to remove, if desired.")
    parser.add_argument("--imgfolder", help="folder with training images.")
    parser.add_argument("--examples_req", type=int, help="number of training examples per class required. Classes with too few examples will be skipped.", required=False)
    parser.add_argument("--split", help="percentage of data used for training.", type=float)
    parser.add_argument("--resize", nargs=2, help="output image size, in px.", type=int)
    parser.add_argument("--outfolder", help="location of output images and CSV files.")
    return parser.parse_args(args)

def run(args):
    with open(args.datacsv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        all_data = [row for row in reader]

        if args.remove_prefix:
            all_data = [row for row in all_data if not row['Name'].startswith(args.remove_prefix)]

        if args.examples_req:
            names = [row['Name'] for row in all_data]
            unique, counts = np.unique(names, return_counts=True)
            unique_n = unique[counts >= args.examples_req]
            all_data = [row for row in all_data if row['Name'] in unique_n]


        #load all images and preprocess
        #NOTE: loading all images into memory - will be fine for now, but could easily become unwieldy.

        all_img_data = []
        for row in all_data:
            imgname = row['Primary Image']
            with Image.open("{}/{}".format(args.imgfolder, row['Primary Image'])) as img:
                all_img_data.append((imgname, row['Name'], img.resize(args.resize)))


        train, test = train_test_split(all_img_data, train_size=args.split)

        #save training and data w/annotations
        #
        out_train = "{}/train".format(args.outfolder)
        out_test = "{}/test".format(args.outfolder)

        try:
            os.makedirs(out_train)
            os.makedirs(out_test)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        #save train data to CSV
        #
        with open("{}/train.csv".format(args.outfolder), 'w') as f:
            trainwriter = csv.writer(f)
            trainwriter.writerow(['image', 'label'])
            for name, label, img in train:
                trainwriter.writerow([name, label])
                #save image
                img.save("{}/{}".format(out_train, name))

        #save test data to CSV
        with open("{}/test.csv".format(args.outfolder), 'w') as f:
            testwriter = csv.writer(f)
            testwriter.writerow(['image', 'label'])
            for name, label, img in test:
                testwriter.writerow([name, label])
                #save image
                img.save("{}/{}".format(out_test, name))


if __name__=="__main__":
    run(parse(argv[1:]))
