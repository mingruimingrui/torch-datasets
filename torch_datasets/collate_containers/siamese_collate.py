import numpy as np

import torch
import torchvision

from ._siamese_collate_configs import make_configs
from ..utils import transforms


class SiameseCollateContainer(object):
    def __init__(self, **kwargs):
        """ Light weight container that contains the collate instructions
        Meant to be picklable for ease of transfer of batch creating instructions
        """
        self.configs = make_configs(**kwargs)
        self.transform_generator = None

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

    def collate_fn(self, sample_group):
        """ Collate fn requires datasets which returns samples as a dict in the following format
        sample = {
            'image' : Image in HWC RGB format as a numpy.ndarray,
            'label' : Class (as an int) image belongs to
        }
        Returns a sample in the following format
        sample = {
            'image' : torch.Tensor Images in NCHW normalized according to pytorch standard
            'label' : torch.Tensor class ids
        }
        """
        # Gather image and label group
        image_group = [sample['image'] for sample in sample_group]
        label_group = [sample['label'] for sample in sample_group]

        # Preprocess individual samples
        for index, image in enumerate(image_group):
            # Augment sample
            if self.configs['allow_transform']:
                image = self._random_transform_entry(image)
            image = self._resize_image(image)
            image_group[index] = image

        # Compile samples into batch
        image_batch = [self._convert_tensor(image) for image in image_group]
        image_batch = torch.stack(image_batch, dim=0)
        label_batch = label_group

        return {
            'image' : image_batch,
            'label' : label_batch
        }
