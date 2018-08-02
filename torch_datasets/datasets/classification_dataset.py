import os
import json
from collections import OrderedDict

import torch.utils.data

from . import _getters, _setters, _editors, _misc


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file=None, root_dir=None):
        """ ClassificationDataset class
        Args
            dataset_file : Path to ClassificationDataset json file
                           (or None if creating new dataset)
            root_dir     : Path to root directory of all image files in dataset
                           (only needed if creating new dataset)
        """
        self.dataset_file = dataset_file
        self.dataset_type = 'classification'

        self.id_to_class_info   = OrderedDict()
        self.name_to_class_info = OrderedDict()
        self.image_infos        = OrderedDict()

        if dataset_file is None:
            self._create_dataset(root_dir)
            print('Created new empty dataset')
        elif os.path.isfile(self.dataset_file):
            print('Existing dataset found, loading dataset')
            self._load_dataset()
            print('Dataset loaded')
        else:
            raise FileNotFoundError('dataset_file "{}" does not exist'.format(dataset_file))

    def _create_dataset(self, root_dir):
        assert os.path.isdir(root_dir), '{} is not a valid path for root_dir'.format(root_dir)
        self.root_dir = root_dir
        self.next_image_id = 0
        self.all_image_index = self.list_all_image_index()

    def _load_dataset(self):
        with open(self.dataset_file, 'r') as f:
            data = json.load(f)

        # Save root dir
        self.root_dir = data['root_dir']

        # Ensure that dataset is a classification dataset
        assert data['dataset_type'] == 'classification', 'Dataset loaded is "{}" type instead of "classification"'.format(data['dataset_type'])

        # Retrieve object class information
        for class_info in data['classes']:
            self.id_to_class_info[class_info['id']]     = class_info
            self.name_to_class_info[class_info['name']] = class_info

        # Retrieve image information
        for image_info in data['images']:
            self.image_infos[image_info['id']] = image_info
        self.next_image_id = max(self.image_infos.keys()) + 1
        self.all_image_index = self.list_all_image_index()

    def save_dataset(self, dataset_file=None, force_overwrite=False):
        """ Save ClassificationDataset to a json file
        Args
            dataset_file    : Path to ClassificationDataset json file (or None if saving to same file dataset is loaded from)
            force_overwrite : Flag to raise if overwriting over existing dataset file
        """
        if dataset_file is not None:
            self.dataset_file = dataset_file

        # Initialize dict
        json_dataset = OrderedDict()

        # Save dataset info
        json_dataset['root_dir'] = self.root_dir
        json_dataset['dataset_type'] = self.dataset_type
        json_dataset['classes'] = list(self.name_to_class_info.values())
        json_dataset['images'] = list(self.image_infos.values())

        # Save dataset into json file
        if (not os.path.isfile(self.dataset_file)) or force_overwrite:
            print('Saving dataset as an annotation file, this can take a while')
            with open(self.dataset_file, 'w') as f:
                json.dump(json_dataset, f)
            print('Dataset saved')
        else:
            raise FileExistsError('Dataset not saved as it already exists, consider overwriting')

    ###########################################################################
    #### Dataset misc functions

    _prepare_classes   = _misc._prepare_classes
    _prepare_bbox      = _misc._prepare_bbox
    _populate_image_hw = _misc._populate_image_hw

    ###########################################################################
    #### Dataset getter and loaders

    get_dataset_file = _getters.get_dataset_file
    get_size         = _getters.get_size
    get_root_dir     = _getters.get_root_dir
    get_num_classes  = _getters.get_num_classes

    name_to_label = _getters.name_to_label
    label_to_name = _getters.label_to_name

    list_classes         = _getters.list_classes
    list_all_image_index = _getters.list_all_image_index

    get_image              = _getters.get_image
    get_image_info         = _getters.get_image_info
    get_image_path         = _getters.get_image_path
    get_image_url          = _getters.get_image_url
    get_image_height       = _getters.get_image_height
    get_image_width        = _getters.get_image_width
    get_image_aspect_ratio = _getters.get_image_aspect_ratio
    get_image_bbox         = _getters.get_image_bbox
    get_image_class_ids    = _getters.get_image_class_ids

    ###########################################################################
    #### Dataset setters

    set_classes = _setters.set_classes

    def set_image(
        self,
        image_path=None,
        image_url=None,
        image_id=None,
        classes=None,
        bbox=None,
        height=None,
        width=None,
        force_overwrite=False
    ):
        """ Sets an image entry in the dataset

        Required variables:
            image_path/image_url (atleast 1 required)
            classes

        Args
            image_path      : The path to the locally stored image relative to root_dir
            image_url       : The http public url to the image
            image_id        : An integer to use for the image id
            classes         : Classes assigned to this image, can be a string, integer, or list of both (only for classification task)
            Eg. classes can be ['asian', 'teen', 'male']
                Then image classes can be ['asian', ['teen', 2], [0, 2], 1]   (Here we just have 4 images)
                1st image will have only the label 'asian'
                2nd image will have the labels 'teen' and 'male'
                3rd image will have the labels 'asian' and 'male'
                4th image will have only the label 'teen'
            bbox            : The bounding box to apply to image (does not apply for detection datasets)
            height          : The image pixel-wise height
            width           : The image pixel-wise width
            force_overwrite : Flag to trigger the overwrite of image at image_id
        Returns
            image info (Dataset object will also be updated with this new image info)
        """
        image_info = _setters.set_image(
            self,
            image_path=image_path,
            image_url=image_url,
            image_id=image_id,
            height=height,
            width=width,
            force_overwrite=force_overwrite
        )
        image_id = image_info['id']

        # Prepare image classes and bbox
        class_ids = self._prepare_classes(classes)
        bbox      = self._prepare_bbox(bbox, image_id)

        # Update image_info
        image_info['bbox']      = bbox
        image_info['class_ids'] = class_ids

        return image_info

    ###########################################################################
    #### Dataset editor

    ###########################################################################
    #### torch.utils.data.Dataset functions

    def __len__(self):
        return len(self.all_image_index)

    def __getitem__(self, idx):
        image_id = self.all_image_index[idx]
        return {
            'image' : self.get_image(image_id),
            'bbox'  : self.get_image_bbox(image_id),
            'labels': self.get_image_class_ids(image_id)
        }
