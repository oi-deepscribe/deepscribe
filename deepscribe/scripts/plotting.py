
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
