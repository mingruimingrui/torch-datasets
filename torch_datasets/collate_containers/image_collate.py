import collections
import numpy as np

import torch
import torchvision

from ._image_collate_configs import make_configs
from ..utils import transforms


class ImageCollateContainer(object):
    def __init__(self, **kwargs):
        """ Light weight container that contains the collate instructions
        Meant to be picklable for ease of transfer of batch creating instructions
        """
        self.configs = make_configs(**kwargs)

    def _make_random_transformation(self):
        return transforms.random_transform(
            min_rotation    = self.configs['min_rotation'],
            max_rotation    = self.configs['max_rotation'],
            min_translation = self.configs['min_translation'],
            max_translation = self.configs['max_translation'],
            min_shear       = self.configs['min_shear'],
            max_shear       = self.configs['max_shear'],
            min_scaling     = self.configs['min_scaling'],
            max_scaling     = self.configs['max_scaling'],
            flip_x_chance   = self.configs['flip_x_chance'],
            flip_y_chance   = self.configs['flip_y_chance']
        )

    def _random_transform_entry(self, image):
        transformation = self._make_random_transformation()
        transformation = transforms.adjust_transform_for_image(transformation, image)
        image = transforms.apply_transform_to_image(transformation, image)
        return image

    def _resize_image(self, image):
        image = transforms.resize_image_2(
            image,
            width=self.configs['image_width'],
            height=self.configs['image_height'],
            stretch_to_fill=self.configs['stretch_to_fill']
        )
        return image

    def _convert_tensor(self, image):
        image = transforms.preprocess_img(image)
        return image

    def collate_fn(self, group):
        """ Collate function that collates a list of images into a preprocessed tensor
        Args
            group : List of dicts or numpy.ndarray, each entry can be of either
                    - A numpy.ndarray image in the HWC, RGB format
                    - A dict in the
                      {
                          'image' : numpy.ndarray image in the HWC, RGB format,
                          'bbox'  : list-like in the format (x1, y1, x2, y2) or None
                      }

            If you do not wish to perform cropping, you can do any one of the following
            - Just provide a list of numpy.ndarray
            - Do not provide the 'bbox' key in the dict entry
            - Set 'bbox' to be None

        Returns
            A torch.Tensor in the format NCHW normalized according to pytorch standard
        """
        image_batch = [None] * len(group)

        # Preprocess individual samples
        for index, entry in enumerate(group):
            # Identify entry type
            if isinstance(entry, np.ndarray):
                image = entry
                bbox = None
            elif isinstance(entry, collections.Mapping):
                image = entry['image']
                bbox = entry['bbox'] if 'bbox' in entry else None
            else:
                raise ValueError('{} is an unsupported entry type'.format(type(batch)))

            # Crop sample
            if bbox is not None:
                image = image[bbox[1]:bbox[3], bbox[0]:bbo[2]]

            # Augment sample
            if self.configs['allow_transform']:
                image = self._random_transform_entry(image)

            image = self._resize_image(image)
            image = self._convert_tensor(image)
            image_batch[index] = image

        return torch.stack(image_batch, dim=0)
