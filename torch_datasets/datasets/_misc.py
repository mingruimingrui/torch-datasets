from __future__ import division

import os
import warnings
from ..utils.image_io import get_image_size, read_image_url


def _prepare_classes(self, classes):
    if classes is None:
        return []

    if not isinstance(classes, list):
        classes = [classes]

    for i, c in enumerate(classes):
        if c in self.id_to_class_info:
            classes[i] = c
        elif c in self.name_to_class_info:
            classes[i] = self.name_to_class_info[c]['id']
        else:
            raise Exception('The class {} is invalid'.format(c))

    return classes

def _prepare_bbox(self, bbox, image_id):
    """ Checks and converts a bbox to (4,) """
    if bbox is None:
        return None

    image_height = self.get_image_height(image_id)
    image_width  = self.get_image_width(image_id)

    # Convert to (4,)
    if len(bbox) == 2:
        assert len(bbox[0]) == 2, 'Invalid bbox shape must be either (4,) or (2, 2) but not {}'.format(bbox)
        assert len(bbox[1]) == 2, 'Invalid bbox shape must be either (4,) or (2, 2) but not {}'.format(bbox)
        bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
    assert len(bbox) == 4, 'Invalid bbox shape must be either (4,) or (2, 2) but not {}'.format(bbox)

    # Check valid
    if (bbox[2] <= bbox[0])  or (bbox[3] <= bbox[1]):
        warnings.warn('Invalid bbox {}'.format(bbox))
    if (bbox[0] < 0) or (bbox[1] < 0):
        warnings.warn('Invalid bbox {}'.format(bbox))
    if (bbox[2] > image_width) or (bbox[3] > image_height):
        warnings.warn('Invalid bbox {}, image height is {}, image width is {}'.format(
            bbox, image_width, image_height))

    return bbox

def _populate_image_hw(self, image_id):
    """ Gets the image height and width and sets the values into image_infos """
    image_info = self.image_infos[image_id]

    if image_info['image_path'] is not None:
        width, height = get_image_size(os.path.join(self.root_dir, image_info['image_path']))
    else:
        height, width = read_image_url(image_info['image_url']).shape[:2]

    image_info['height']       = height
    image_info['width']        = width
    image_info['aspect_ratio'] = width / height

    self.image_infos[image_id] = image_info
