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

    def collate_fn(self, image_group):
        """ Collate function that collates a list of images into a preprocessed tensor
        Args
            image_group : List of numpy.ndarray images in the HWC, RGB format
        Returns
            A torch.Tensor in the format NCHW normalized according to pytorch standard
        """
        image_batch = [None] * len(image_group)

        # Preprocess individual samples
        for index, image in enumerate(image_group):
            # Augment sample
            if self.configs['allow_transform']:
                image = self._random_transform_entry(image)
            image = self._resize_image(image)
            image = self._convert_tensor(image)
            image_batch[index] = image

        return torch.stack(image_batch, dim=0)
