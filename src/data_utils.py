#data processing utilities for OCHRE data.

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy

def summarize_data(csvloc):
    '''
    Reads and prints a summary of the cuneiform sign data contained in csvfile.
    '''
    with open(csvloc, 'r') as f:
        reader = csv.DictReader(f)
        names = [row['Name'] for row in reader if not row['Name'].startswith("PFS")]

        print("Number of labeled images in data set: {}".format(len(names)))

        unique, counts = np.unique(names, return_counts=True)

        print("unique classes: {}".format(len(unique)))

        min_examples = np.argmin(counts)
        print("class with smallest number of examples: {} with {}".format(unique[min_examples], counts[min_examples]))
        max_examples = np.argmax(counts)
        print("class with largest number of examples: {} with {}".format(unique[max_examples], counts[max_examples]))

        print("classes with fewer than 10 training examples: {}".format(len(counts[counts < 10])))

        print("median examples per class: {}".format(np.median(counts)))
        print("mean examples per class: {}, std: {}".format(np.mean(counts), np.std(counts)))

        print("mean examples per class, excluding classes w/fewer than 10 examples: {}, std: {}".format(np.mean(counts[counts >= 10]), np.std(counts[counts >= 10])))

        #saving histogram of data distribution
        plt.figure()
        plt.hist(counts, bins=20)
        plt.xlabel("number of examples per sign")
        plt.ylabel("unique sign/class count")
        plt.title("Distribution of Sign Image Data")
        plt.savefig("data/processed/analysis/hist.png")


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
