from __future__ import division

import gzip
import logging
import operator
import os
import struct
import urlparse

import numpy as np

import mldata.util
from mldata.dataset import Dataset
from mldata.dataset import DatasetGroup


logger = logging.getLogger(__name__)


class MNIST(DatasetGroup):
    """The MNIST database of handwritten digits.

    Homepage: http://yann.lecun.com/exdb/mnist/

    Images are 28x28 grayscale images in the range [0, 1].
    """

    base_url = 'http://yann.lecun.com/exdb/mnist/'

    data_files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz',
            }

    def __init__(self, path=None):
        DatasetGroup.__init__(self, 'mnist', path=path)
        self._load_datasets()

    def download(self):
        for filename in self.data_files.values():
            url = urlparse.urljoin(self.base_url, filename)
            dest = self.get_path(filename)
            mldata.util.maybe_download(url, dest)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images = self._read_images(abspaths['train_images'])
        train_labels = self._read_labels(abspaths['train_labels'])
        test_images = self._read_images(abspaths['test_images'])
        test_labels = self._read_labels(abspaths['test_labels'])
        self.train = Dataset(train_images, train_labels)
        self.test = Dataset(test_images, test_labels)

    def _read_datafile(self, path, expected_dims):
        """Helper function to read a file in IDX format."""
        base_magic_num = 2048
        with gzip.GzipFile(path) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + expected_dims
            if magic_num != expected_magic_num:
                raise ValueError('Incorrect MNIST magic number (expected '
                                 '{}, got {})'
                                 .format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * expected_dims,
                                 f.read(4 * expected_dims))
            buf = f.read(reduce(operator.mul, dims))
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(*dims)
            return data

    def _read_images(self, path):
        """Read an MNIST image file."""
        return self._read_datafile(path, 3) / 255

    def _read_labels(self, path):
        """Read an MNIST label file."""
        return self._read_datafile(path, 1)
