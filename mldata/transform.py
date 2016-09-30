import numpy as np
import skimage.transform


class ImageTransformer(object):

    def __init__(self, image_size=None):
        self.image_size = image_size

    def preprocess_batch(self, batch):
        preprocessed = []
        for im in batch:
            preprocessed.append(self.preprocess(im))
        return np.stack(preprocessed, axis=0)

    def preprocess(self, im):
        if self.image_size is not None:
            im = skimage.transform.resize(im, self.image_size,
                                          preserve_range=True)
        return im
