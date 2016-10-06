from __future__ import division

import gzip
import logging
import os
import urlparse

import numpy as np

from mldata import util
from mldata import dataset


logger = logging.getLogger(__name__)


@dataset.register_dataset('usps')
class USPS(dataset.DatasetGroup):
    """USPS handwritten digits.

    Homepage: http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    def __init__(self, path=None):
        dataset.DatasetGroup.__init__(self, 'usps', path=path)
        self._load_datasets()

    def download(self):
        for filename in self.data_files.values():
            url = urlparse.urljoin(self.base_url, filename)
            dest = os.path.join(self.path, filename)
            util.maybe_download(url, dest)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images, train_labels = self._read_datafile(abspaths['train'])
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.train = dataset.Dataset(train_images, train_labels)
        self.test = dataset.Dataset(test_images, test_labels)

    def _read_datafile(self, path):
        """Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.float32)
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels
