import torch
import torchvision

from ._detection_collate_configs import make_configs
from ..utils import transforms


class DetectionCollateContainer(object):
    def __init__(self, **kwargs):
        """ Light weight container that contains the collate instructions
        Meant to be picklable for ease of transfer of batch creating instructions
        """
        self.configs = make_configs(**kwargs)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.transform_generator = None
        if self.config['allow_transform']:
            self.transform_generator = self._make_transform_generator()

    def _make_transform_generator(self):
        return random_transform_generator(
            min_rotation    = self.configs['min_rotation'],
            max_rotation    = self.configs['max_rotation'],
            min_translation = self.configs['min_translation'],
            max_translation = self.configs['max_translation'],
            min_shear       = self.configs['min_shear'],
            max_shear       = self.configs['max_shear'],
            min_scaling     = self.configs['min_scaling'],
            max_scaling     = self.configs['max_scaling'],
            flip_x_chance   = self.configs['flip_x_chance'],
            flip_y_chance   = self.configs['flip_y_chance'],
        )

    def random_transform_entry(self, image, annotations):
        transformation = adjust_transform_for_image(next(self.transform_generator), image)

        # Transform image and annotations
        image = apply_transform_to_image(transformation, image)
        annotations = annotations.copy()
        for index in range(annotations.shape[0]):
            annotations[index, :4] = apply_transform_aabb(transformation, annotations[index, :4])

        return image, annotations

    def resize_image(self, image, annotations):
        image, scale = transforms.resize_image_1(
            image,
            min_side=self.configs['image_min_side'],
            max_side=self.configs['image_max_side']
        )
        annotations[:, :4] = annotations[:, :4] * scale
        return image, annotations

    def filter_annotations(self, image, annotations):
        assert isinstance(annotations, np.ndarray)

        # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
        invalid_indices = np.where(
            (annotations[:, 2] <= annotations[:, 0]) |
            (annotations[:, 3] <= annotations[:, 1]) |
            (annotations[:, 0] < 0) |
            (annotations[:, 1] < 0) |
            (annotations[:, 2] > image.shape[1]) |
            (annotations[:, 3] > image.shape[0])
        )[0]

        if len(invalid_indices):
            annotations = np.delete(annotations, invalid_indices, axis=0)

        return image, annotations

    def collate_fn(self, sample_group):
        """ Collate fn requires datasets which returns samples as a dict in the following format
        sample = {
            'image'       : Image in HWC RGB format as a numpy.ndarray,
            'annotations' : Annotations of shape (num_annotations, 5) also numpy.ndarray
                - each row will represent 1 detection target of the format
                (x1, y1, x2, y2, class_id)
        }
        Returns a sample in the following format
        sample = {
            'image'          : torch.Tensor Images in NCHW normalized according to pytorch standard
            'annotations'    : list of torch.Tensor of shape (N, num_anchors, 5)
                               Number of objects in list corresponds to batch size
        }
        """
        # Gather image and annotations group
        image_group       = [sample['image'] for sample in sample_group]
        annotations_group = [sample['annotations'] for sample in sample_group]

        # Preprocess individual samples
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # Augment sample
            if self.configs['allow_transform']:
                image, annotations = self.random_transform_entry(image, annotations)

            image, annotations = self.resize_image(image, annotations)
            image, annotations = self.filter_annotations(image, annotations)

            image_group[index] = image
            annotations_group[index] = annotations

        # Compile samples into Tensor batches
        max_image_hw = tuple(max(image.shape[x] for image in image_group) for x in range(2))
        image_batch = []
        annotations_batch = []

        for image, annotations in zip(image_group, annotations_group):
            # Perform normalization on image and convert to tensor
            image = transforms.pad_img_to(image, max_image_hw)
            image = self.to_tensor(image)
            image = self.normalize(image)

            # Convert annotations to tensors
            annotations = torch.Tensor(annotations)

            image_batch.append(image)
            annotations_batch.append(annotations)

        # Stack image batches only as annotations batch can be differently sized
        image_batch = torch.stack(image_batch, dim=0)

        return {
            'image'       : image_batch,
            'annotations' : annotations_batch
        }
