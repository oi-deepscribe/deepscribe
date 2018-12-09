# Python command line utility.

from argparse import ArgumentParser
from sys import argv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def parse(args):
    """
    Parse stuff.
    """
    parser = ArgumentParser()
    parser.add_argument("--npz", help="NPZ archive with images")
    parser.add_argument("--nsamples", help="number of samples to plot for each class", type=int)
    parser.add_argument("--out", help="output prefix.")
    return parser.parse_args(args)

def run(args):
    """
    Do stuff.
    """

    loaded = np.load(args.npz)

    imgs = loaded['imgs']
    # integer labels
    labels = loaded['labels']

    num_classes = np.max(labels) + 1
    print("{} classes total".format(num_classes))

    for i in range(num_classes):
        res = np.where(labels == i)
        inds = res[0]
        # get sample from valid indices
        samp_inds = np.random.choice(inds, size=args.nsamples)

        sz = int(np.sqrt(args.nsamples))

        fig = plt.figure(1, (4., 4.))

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(sz + 1, sz + 1),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

        for j in range(args.nsamples):
            grid[j].imshow(imgs[samp_inds[j], :, :,0])
        plt.title("samples from class {}".format(i))
        plt.savefig("{}class_{}.png".format(args.out, i))
        plt.close()

if __name__=="__main__":
    run(parse(argv[1:]))
