import os
import numpy as np

import mldata.util


class DatasetGroup(object):

    def __init__(self, name, path=None):
        self.name = name
        if path is None:
            path = mldata.util.get_path(self.name)
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.download()

    def get_path(self, *args):
        """Build a path using this dataset group's root directory."""
        return os.path.join(self.path, *args)

    def download(self):
        """Download the dataset if it does not reside on disk."""
        raise NotImplementedError


class Dataset(object):

    def __init__(self, images, labels, shuffle=True):
        if len(images) != len(labels):
            raise TypeError('Different number of images and labels')
        self.images = images
        self.labels = labels
        self.shuffle = shuffle
        self._mask = np.arange(len(self))
        self._reset_inds()
        self.epochs_completed = 0

    def __len__(self):
        return len(self.labels)

    def mask(self, inds):
        """Subselect only a portion of the dataset.

        If inds is None, the mask is reset to include the entire dataset.
        """
        if inds is None:
            self._mask = np.arange(len(self))
        else:
            self._mask = inds
        self._reset_inds()

    def next_batch(self, n):
        """Return an (images, labels) tuple containing the next batch."""
        inds = np.array([], dtype=np.int)
        while len(inds) < n:
            remaining = n - len(inds)
            inds = np.concatenate((inds, self._inds[:remaining]))
            self._inds = self._inds[remaining:]
            if len(self._inds) == 0:
                # completed full pass through dataset
                self._reset_inds()
                self.epochs_completed += 1
        images = self.images[inds]
        labels = self.labels[inds]
        return images, labels

    def _reset_inds(self):
        """Reset the internal list of indices not yet sampled this epoch.

        This will also shuffle the list of indices if necessary.
        """
        self._inds = self._mask.copy()
        if self.shuffle:
            np.random.shuffle(self._inds)
