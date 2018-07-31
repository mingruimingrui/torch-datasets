from __future__ import division

import math
import cv2
import numpy as np
import torch

from .collections import AttrDict
from ._transforms import (
    change_transform_origin,
    random_transform,
    random_transform_generator
)


TORCH_IMG_MEAN = [0.485, 0.456, 0.406]
TORCH_IMG_STD  = [0.229, 0.224, 0.225]


def preprocess_img(img):
    """ Preprocess an image based on torch convention """
    # Make a copy of img as array
    img = np.array(img)

    # Convert into tensor
    img = torch.Tensor(img).permute(2, 0, 1) / 255.0

    # Normalize
    for t, m, s in zip(img, TORCH_IMG_MEAN, TORCH_IMG_STD):
        t.sub_(m).div_(s)

    return img


def preprocess_img_inv(img):
    """ Unpreprocess an image based on torch convention and returns an array """
    img = img.data.numpy().copy()

    img[0] = img[0] * TORCH_IMG_STD[0] + TORCH_IMG_MEAN[0]
    img[1] = img[1] * TORCH_IMG_STD[1] + TORCH_IMG_MEAN[1]
    img[2] = img[2] * TORCH_IMG_STD[2] + TORCH_IMG_MEAN[2]
    img = img.transpose(1, 2, 0) * 255.0

    return img.round().astype('uint8')


def resize_image_1(img, min_side=800, max_side=1333):
    """ Resizes a numpy.ndarray of the format HWC such that the
    smallest side >= min_side and largest side <= max_side
    In the event that scaling is not possible to meet both conditions, only
    largest side <= max_side will be satisfied
    """
    rows, cols = img.shape[:2]

    smallest_side = min(rows, cols)
    largest_side = max(rows, cols)

    # Don't resize if size acceptable
    if smallest_side >= min_side and largest_side <= max_side:
        return img.copy(), 1.0

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def resize_image_2(img, width=224, height=224, stretch_to_fill=True):
    if stretch_to_fill:
        # If to fill to maximum width and height
        img = cv2.resize(img, dsize=(width, height))

    else:
        # Else we identify which axis to scale against
        img_height, img_width = img.shape[:2]

        scale_height = height / img_height
        scale_width  = width  / img_width

        # Get the smaller scale
        scale = min(scale_height, scale_width)

        # resize the image with the computed scale
        img = cv2.resize(img, None, fx=scale, fy=scale)

        # Pad image
        img = pad_img_to(img, (height, width), location='center')

    return img


def pad_img_to(img, target_hw, location='upper-left', mode='constant'):
    """ Takes an numpy.ndarray image of the format HWC and pads it to the target_hw

    Args
        img       : numpy.ndarray image of the format HWC or HW
        target_hw : target height width (list-like of size 2)
        location  : location of original image after padding, option of 'upper-left' and 'center'
        mode      : mode of padding in https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html
    Returns
        padded image

    The original image will be placed in the top left corner
    """
    if len(img.shape) == 3:
        pad = [None, None, (0, 0)]
    else:
        pad = [None, None]

    if location == 'upper-left':
        for i in range(2):
            pad[i] = (0, target_hw[i] - img.shape[i])

    elif location == 'center':
        for i in range(2):
            excess = target_hw[i] - img.shape[i]
            x1 = math.ceil(excess / 2)
            x2 = excess - x1
            pad[i] = (x1, x2)

    else:
        raise ValueError('{} is not a valid location argument'.format(location))

    return np.pad(img, pad, mode=mode)


def adjust_transform_for_image(transform, image, relative_translation=True):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


def create_transform_parameters(
    fill_mode            = 'nearest',
    interpolation        = 'linear',
    cval                 = 0,
    data_format          = None,
    relative_translation = True,
):
    """ Creates a dictionary to store parameters containing information on
    method to apply transformation to an image

    # Arguments
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        data_format:           Same as for keras.preprocessing.image_transform.apply_transform
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    # Apply processing to input arguments
    if data_format is None:
        data_format = 'channels_last'

    if data_format == 'channels_first':
        channel_axis = 0
    elif data_format == 'channels_last':
        channel_axis = 2
    else:
        raise ValueError("invalid data_format, expected 'channels_first' or 'channels_last', got '{}'".format(data_format))

    if fill_mode == 'constant':
        cv_border_mode = cv2.BORDER_CONSTANT
    if fill_mode == 'nearest':
        cv_border_mode = cv2.BORDER_REPLICATE
    if fill_mode == 'reflect':
        cv_border_mode = cv2.BORDER_REFLECT_101
    if fill_mode == 'wrap':
        cv_border_mode = cv2.BORDER_WRAP

    if interpolation == 'nearest':
        cv_interpolation = cv2.INTER_NEAREST
    if interpolation == 'linear':
        cv_interpolation = cv2.INTER_LINEAR
    if interpolation == 'cubic':
        cv_interpolation = cv2.INTER_CUBIC
    if interpolation == 'area':
        cv_interpolation = cv2.INTER_AREA
    if interpolation == 'lanczos4':
        cv_interpolation = cv2.INTER_LANCZOS4

    # Create attribute dict to store parameters
    _p = AttrDict(
        fill_mode=fill_mode,
        interpolation=interpolation,
        cval=cval,
        relative_translation=relative_translation,
        data_format=data_format,
        channel_axis=channel_axis,
        cv_border_mode=cv_border_mode,
        cv_interpolation=cv_interpolation
    )
    _p.immutable(True)

    return _p


DEFAULT_TRANSFORM_PARAMETERS = create_transform_parameters()


def apply_transform_to_image(matrix, image, params=DEFAULT_TRANSFORM_PARAMETERS):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Parameters:
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    if params.channel_axis != 2:
        image = np.moveaxis(image, params.channel_axis, 2)

    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cv_interpolation,
        borderMode  = params.cv_border_mode,
        borderValue = params.cval,
    )

    if params.channel_axis != 2:
        output = np.moveaxis(output, 2, params.channel_axis)
    return output


def apply_transform_aabb(transform, aabb):
    """ Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    # Arguments
        transform: The transformation to apply.
        x1:        The minimum X value of the AABB.
        y1:        The minimum y value of the AABB.
        x2:        The maximum X value of the AABB.
        y2:        The maximum y value of the AABB.
    # Returns
        The new AABB as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def apply_transform_xy(transform, xy):
    """ Apply a transformation to a set of points.

    The inputs should be a set of points in the format [(x1, y1), (x2, y2), ...]
    The result is the transformed set of points of the same shape as the input.

    # Arguments
        transform: The transformation to apply.
        xy       : A set of points in the implied shape of (n, 2)
    # Returns
        The new XY as tuple
    """
    # Get number of points, store as n
    n = len(xy)

    # _xy is the transformed xy in inplied 3d vector form
    _xy = np.hstack([xy, np.ones([n,1])])

    # Apply transformation to all points
    points = transform.dot(_xy.T)

    return points[:2].T
