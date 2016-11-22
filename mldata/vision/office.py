from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import skimage
from skimage.transform import resize
from skimage.io import imread, imsave

from mldata import dataset


logger = logging.getLogger(__name__)


@dataset.register_dataset('office')
class Office(dataset.DatasetGroup):

    classes = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle',
               'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer',
               'file_cabinet', 'headphones', 'keyboard', 'laptop_computer',
               'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug',
               'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker',
               'stapler', 'tape_dispenser', 'trash_can']

    domains = ['amazon', 'webcam', 'dslr']

    def __init__(self, path=None, image_size=(256, 256)):
        dataset.DatasetGroup.__init__(self, 'office', path=path)
        self.image_size = image_size
        self._load_datasets()

    def _cache_path(self, *args):
        return os.path.join('/tmp', 'mldata', self.name,
                            '{}-{}'.format(*self.image_size), *args)

    def download(self):
        pass

    def _load_datasets(self):
        for domain in self.domains:
            images = []
            labels = []
            for dirpath, dirnames, filenames in os.walk(self.get_path(domain)):
                for filename in filenames:
                    if not filename.endswith('.jpg'):
                        continue
                    label = os.path.basename(dirpath)
                    labels.append(self.classes.index(label))
                    cachepath = self._cache_path(domain, label, filename)
                    if os.path.exists(cachepath):
                        im = imread(cachepath)
                        im = skimage.img_as_float(im)
                        images.append(im)
                    else:
                        fullpath = os.path.join(dirpath, filename)
                        im = resize(imread(fullpath), self.image_size)
                        cachedir = os.path.dirname(cachepath)
                        if not os.path.exists(cachedir):
                            os.makedirs(cachedir)
                        imsave(cachepath, im)
                        images.append(im)
            images = np.stack(images, axis=0)
            labels = np.array(labels)
            setattr(self, domain, dataset.Dataset(images, labels))
