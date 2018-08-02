""" File to store all Dataset setter functions """
from __future__ import division


def set_classes(self, classes):
    """ Sets the classes for this given dataset
    Args
        classes: List of strings representing all classes
    """
    assert len(self.id_to_class_info) == 0, 'Classes appear to already have been set in this dataset, please create new dataset'

    for class_id, class_name in enumerate(classes):
        class_info = {
            'id'   : class_id,
            'name' : class_name
        }
        self.id_to_class_info[class_id]     = class_info
        self.name_to_class_info[class_name] = class_info

    print('Classes saved')

def set_image(
    self,
    image_path=None,
    image_url=None,
    image_id=None,
    height=None,
    width=None,
    force_overwrite=False
):
    """ Sets an image entry in the dataset

    Required variables:
        image_path/image_url (atleast 1 required)

    Args
        image_path      : The path to the locally stored image relative to root_dir
        image_url       : The http public url to the image
        image_id        : An integer to use for the image id
        bbox            : The bounding box to apply to image (does not apply for detection datasets)
        height          : The image pixel-wise height
        width           : The image pixel-wise width
        force_overwrite : Flag to trigger the overwrite of image at image_id
    Returns
        image info (Dataset object will also be updated with this new image info)
    """
    assert (image_url is not None) or (image_path is not None), 'Atleast one of image path or image url must be provided'

    # Identify image id
    if image_id is None:
        image_id = self.next_image_id
        self.next_image_id += 1
    else:
        assert isinstance(image_id, int), 'Image id if provided must be an integer, got {}'.format(type(image_id))
        assert (image_id not in self.image_infos) or force_overwrite, 'Image id {} already exists, consider overwrite'.format(image_id)
        self.next_image_id = max(self.next_image_id, image_id) + 1

    # Collect image info and gather into dict object
    image_info = {
        'id'           : image_id,
        'image_path'   : image_path,
        'image_url'    : image_url,
        'width'        : width,
        'height'       : height,
        'aspect_ratio' : width / height if (width is not None and height is not None) else None
    }

    # Store all required info
    self.image_infos[image_id] = image_info
    self.all_image_index += [image_id]

    return image_info

def set_ann(
    self,
    image_id,
    bbox,
    class_name=None,
    class_id=None,
    segmentation=None
):
    """ Sets a single image detection annotation, set_classes has to be ran in advanced

    Args
        image_id  : Image id associated to this detection
        bbox      : Bounding box for detection
        ann_class : Class of this detection instance
    """
    assert (class_name is not None) or (class_id is not None), 'Either class_name or class_id must be present'
    if class_name is not None:
        assert class_name in self.name_to_class_info
        class_id = self.name_to_label(class_name)
    else:
        assert class_id in self.id_to_class_info
        class_name = self.label_to_name(class_id)

    # prepare bbox
    bbox = self._prepare_bbox(bbox, image_id)

    # Prepare ann_info
    ann_id = len(self.ann_infos)
    ann_info = {
        'id'          : ann_id,
        'image_id'    : image_id,
        'bbox'        : bbox,
        'class_id'    : class_id,
        'class_name'  : class_name,
        'segmentation': segmentation
    }

    # Store ann info
    self.ann_infos[ann_id] = ann_info
    self.img_to_ann[image_id].append(ann_id)

    return ann_info
