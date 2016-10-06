from __future__ import division

import urlparse

import numpy as np
from scipy.io import loadmat

from mldata import dataset
from mldata import util


@dataset.register_dataset('svhn_cropped')
class CroppedSVHN(dataset.DatasetGroup):
    """The Street View House Numbers Dataset.

    This DatasetGroup corresponds to format 2, which consists of center-cropped
    digits.

    Homepage: http://ufldl.stanford.edu/housenumbers/

    Images are 32x32 RGB images in the range [0, 1].
    """

    base_url = 'http://ufldl.stanford.edu/housenumbers/'

    data_files = {
            'train': 'train_32x32.mat',
            'test': 'test_32x32.mat',
            'extra': 'extra_32x32.mat',
            }

    def __init__(self, path=None, train_on_extra=False):
        dataset.DatasetGroup.__init__(self, 'svhn_cropped', path=path)
        self.train_on_extra = train_on_extra
        self._load_datasets()

    def download(self):
        for filename in self.data_files.values():
            url = urlparse.urljoin(self.base_url, filename)
            dest = self.get_path(filename)
            util.maybe_download(url, dest)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_mat = loadmat(abspaths['train'])
        train_images = train_mat['X'].transpose((3, 0, 1, 2))
        train_labels = train_mat['y'].squeeze()
        if self.train_on_extra:
            extra_mat = loadmat(abspaths['extra'])
            train_images = np.vstack((train_images,
                                      extra_mat['X'].transpose((3, 0, 1, 2))))
            train_labels = np.concatenate((train_labels,
                                           extra_mat['y'].squeeze()))
        train_images = train_images.astype(np.float32) / 255
        test_mat = loadmat(abspaths['test'])
        test_images = test_mat['X'].transpose((3, 0, 1, 2))
        test_images = test_images.astype(np.float32) / 255
        test_labels = test_mat['y'].squeeze()
        self.train = dataset.Dataset(train_images, train_labels)
        self.test = dataset.Dataset(test_images, test_labels)
