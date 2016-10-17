import os
import numpy as np
import skimage
import skimage.io

import mldata.util
from .transform import ImageTransformer


# Registry to refer to datasets by their name
_registry = {}

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

    def get_image(self, i):
        return self.images[i]

    def mask(self, inds):
        """Subselect only a portion of the dataset.

        If inds is None, the mask is reset to include the entire dataset.
        """
        if inds is None:
            self._mask = np.arange(len(self))
        else:
            self._mask = inds
        self._reset_inds()

    def _next_batch_inds(self, n):
        inds = np.array([], dtype=np.int)
        while len(inds) < n:
            remaining = n - len(inds)
            inds = np.concatenate((inds, self._inds[:remaining]))
            self._inds = self._inds[remaining:]
            if len(self._inds) == 0:
                # completed full pass through dataset
                self._reset_inds()
                self.epochs_completed += 1
        return inds

    def next_batch(self, n):
        """Return an (images, labels) tuple containing the next batch."""
        inds = self._next_batch_inds(n)
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


class ImageDataset(Dataset):

    def __init__(self, image_paths, labels, image_size, shuffle=True):
        Dataset.__init__(self, image_paths, labels, shuffle=shuffle)
        self.transformer = ImageTransformer(image_size=image_size)

    def get_image(self, i):
        im = skimage.io.imread(self.images[i])
        im = skimage.img_as_float(im)
        return self.transformer.preprocess(im)

    def next_batch(self, n):
        inds = self._next_batch_inds(n)
        batch = []
        for ind in inds:
            batch.append(self.get_image(ind))
        images = np.stack(batch, axis=0)
        labels = self.labels[inds]
        return images, labels


def register_dataset(name):
    """Decorator to register a dataset under a given name."""
    def decorator(cls):
        if name in _registry:
            raise KeyError('Duplicate name for dataset: {}'.format(name))
        _registry[name] = cls
        return cls
    return decorator


def make_dataset(name, *args, **kwargs):
    """Create a DatasetGroup based on its registered name."""
    if name not in _registry:
        raise KeyError('Dataset not found: {}'.format(name))
    cls = _registry[name]
    return cls(*args, **kwargs)
