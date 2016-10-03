import numpy as np
import skimage.transform


class ImageTransformer(object):

    def __init__(self, image_size=None, grayscale=False):
        self.image_size = image_size
        self.grayscale = grayscale

    def preprocess_batch(self, batch):
        preprocessed = []
        for im in batch:
            preprocessed.append(self.preprocess(im))
        return np.stack(preprocessed, axis=0)

    def rgb2y(self, im):
        """Converts RGB to luminance, following BT.601."""
        if im.shape[2] == 3:
            return (im * [.299, .587, .114]).sum(axis=2)[:, :, np.newaxis]
        else:
            return im

    def preprocess(self, im):
        if self.grayscale:
            im = self.rgb2y(im)
        if self.image_size is not None:
            im = skimage.transform.resize(im, self.image_size,
                                          preserve_range=True)
        return im
