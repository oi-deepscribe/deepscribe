#data processing utilities for OCHRE data.

import csv
import numpy as np
import scipy

def classes_nexamples(csvfile, n):
    '''
    Filters the CSV file to only include classes with greater than or equal to n.
    Returns a list of dicts containing labels and image UUIDs.
    '''

    with open(csvloc, 'r') as f:
        reader = csv.DictReader(f)
        data_pts = [row for row in reader]

        data_pts_sign = [row for row in data_pts if not row['Name'].startswith("PFS")]

        names = [row['Name'] for row in data_pts_sign]

        unique, counts = np.unique(names, return_counts=True)

        print("unique classes: {}".format(len(unique)))

        print("number of classes with >= {} examples: {}".format(n, len(counts[counts >= n])))

        unique_n = unique[counts >= n]

        data_pts_n = [row for row in data_pts if row['Name'] in unique_n]

        print(data_pts_n)


def load_dataset(labels, img_folder, flatten=True):
    '''
    Loads a dataset from a CSV file containing labels and filenames.
    Converts to grayscale by default.
    '''

    with open(labels, 'r') as f:
        reader = csv.DictReader(f)

        data = [row for row in reader]

        imgs = np.array([scipy.ndimage.imread("{}/{}".format(img_folder,row['image']), flatten=flatten) for row in data])

        labels = np.array([row['label'] for row in data])

    print("{} images read ".format(len(labels)))

    #unzipping
    return imgs, labels



if __name__=="__main__":
    csvloc = "data/ochre/imageIndex.csv"
    n = 30
    outfile = "processed/subsampled/{}_or_more.csv".format(n)
    summarize_data(csvloc)
    data = classes_nexamples(csvloc, n)
