import os
import json
from collections import OrderedDict

import torch.utils.data

from . import _getters, _setters, _editors, _misc


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file=None, root_dir=None):
        """ DetectionDataset class
        Args
            dataset_file : Path to DetectionDataset json file
                           (or None if creating new dataset)
            root_dir     : Path to root directory of all image files in dataset
                           (only needed if creating new dataset)
        """
        self.dataset_file = dataset_file
        self.dataset_type = 'detection'

        self.id_to_class_info   = OrderedDict()
        self.name_to_class_info = OrderedDict()
        self.image_infos        = OrderedDict()
        self.ann_infos          = OrderedDict()
        self.img_to_ann         = OrderedDict()

        if dataset_file is None:
            self._create_dataset(root_dir)
            print('Created new empty dataset')
        elif os.path.isfile(self.dataset_file):
            print('Existing dataset found, loading dataset')
            self._load_dataset()
            print('Dataset loaded')
        else:
            raise FileNotFoundError('dataset_file "{}" does not exist'.format(dataset_file))

    ###########################################################################
    #### Dataset load create save

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

        # Ensure that dataset is a detection dataset
        assert data['dataset_type'] == 'detection', 'Dataset loaded is "{}" type instead of "detection"'.format(data['dataset_type'])

        # Retrieve object class information
        for class_id, class_name in enumerate(data['classes']):
            class_info = {
                'id'   : class_id,
                'name' : class_name
            }
            self.id_to_class_info[class_id]     = class_info
            self.name_to_class_info[class_name] = class_info

        # Retrieve image information
        for image_info in data['images']:
            self.image_infos[image_info['id']] = image_info
        self.next_image_id = max(self.image_infos.keys()) + 1
        self.all_image_index = self.list_all_image_index()

        # Config annotation infos such that it is retrievable through annotation id
        for ann_info in data['annotations']:
            self.ann_infos[ann_info['id']] = ann_info

        # Make the img_to_ann dict
        for image_info in data['images']:
            self.img_to_ann[image_info['id']] = []
        for ann_info in data['annotations']:
            self.img_to_ann[ann_info['image_id']].append(ann_info['id'])


    def save_dataset(self, dataset_file=None, force_overwrite=False):
        """ Save DetectionDataset to a json file
        Args
            dataset_file    : Path to DetectionDataset json file (or None if saving to same file dataset is loaded from)
            force_overwrite : Flag to raise if overwriting over existing dataset file
        """
        # Initialize dict
        json_dataset = OrderedDict()

        # Save dataset info
        json_dataset['root_dir'] = self.root_dir
        json_dataset['dataset_type'] = self.dataset_type
        json_dataset['classes'] = list(name_to_class_info.keys())
        json_dataset['images'] = list(self.image_infos.values())
        json_dataset['annotations'] = list(self.ann_infos.values())

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

    _prepare_bbox = _misc._prepare_bbox

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
    list_all_ann_index   = _getters.list_all_ann_index

    get_image              = _getters.get_image
    get_image_info         = _getters.get_image_info
    get_image_path         = _getters.get_image_path
    get_image_url          = _getters.get_image_url
    get_image_height       = _getters.get_image_height
    get_image_width        = _getters.get_image_width
    get_image_aspect_ratio = _getters.get_image_aspect_ratio

    get_ann_info   = _getters.get_ann_info
    get_ann_array  = _getters.get_ann_array

    ###########################################################################
    #### Dataset setters

    set_classes = _setters.set_classes
    set_image   = _setters.set_image
    set_ann     = _setters.set_ann

    ###########################################################################
    #### Dataset editor

    delete_image = _editors.delete_image

    edit_image_path = _editors.edit_image_path
    edit_image_url  = _editors.edit_image_url

    edit_ann_img_id = _editors.edit_ann_img_id
    edit_ann_class  = _editors.edit_ann_class
    edit_ann_bbox   = _editors.edit_ann_bbox

    ###########################################################################
    #### torch.utils.data.Dataset functions

    def __len__(self):
        return len(self.all_image_index)

    def __getitem__(self, idx):
        image_id = self.all_image_index[idx]
        return {
            'image'      : self.get_image(image_id),
            'annotations': self.get_ann_array(image_id)
        }
