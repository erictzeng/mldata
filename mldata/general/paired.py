import numpy as np


class PairedDataset(object):
    """A Dataset-like object that pairs examples from two Datasets based on
    their labels.
    """

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def next_batch(self, n):
        images1, labels = self.dataset1.next_batch(n)
        images2 = []
        for im1, label in zip(images1, labels):
            inds = np.where(self.dataset2.labels == label)[0]
            im2 = self.dataset2.get_image(np.random.choice(inds))
            images2.append(im2)
        images2 = np.stack(images2, axis=0)
        return images1, images2, labels
