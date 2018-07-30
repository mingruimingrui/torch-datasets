""" Collection of functions to transform popular datasets into torch_dataset Datasets """
import os
import json
import tqdm

from .detection_dataset import DetectionDataset


def convert_coco_to_detection_dataset(coco_ann_file, root_image_dir, no_crowd=False):
    """ Converts a coco annotation file to a detection dataset (which can be saved with save_dataset)
    Args
        coco_ann_file  : The annotation file eg. 'XXX/instances_train2017.json'
        root_image_dir : The folder storing all images eg. 'XXX/train2017/'
        no_crowd       : Flag to switch if crowd object should be included
    Returns
        DetectionDataset object containing coco data
    """

    # Load coco data
    with open(coco_ann_file, 'r') as f:
        print('Loading coco annotation file')
        coco_data = json.load(f)

    # Create empty dataset object
    dataset = DetectionDataset(root_dir=root_image_dir)

    # Set classes
    # Also create link for original class id to class name
    print('Setting classes')
    orig_label_to_name = {}
    all_class_names = []
    for category in coco_data['categories']:
        orig_label_to_name[category['id']] = category['name']
        all_class_names.append(category['name'])
    dataset.set_classes(all_class_names)

    # Set images
    for image in tqdm.tqdm(coco_data['images'], desc='Setting images'):
        dataset.set_image(
            image_path=image['file_name'],
            image_url=image['coco_url'],
            image_id=image['id'],
            height=image['height'],
            width=image['width']
        )

    # Set annotations
    for ann in tqdm.tqdm(coco_data['annotations'], desc='Setting annotations'):
        # Convert bbox to x1, y1, x2, y2
        bbox = ann['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        # import pdb; pdb.set_trace()
        if no_crowd and ann['iscrowd'] == 1:
            continue

        dataset.set_ann(
            image_id=ann['image_id'],
            bbox=bbox,
            class_name=orig_label_to_name[ann['category_id']],
            segmentation=ann['segmentation']
        )

    return dataset

# def convert_wider_to_detection_dataset(wider_ann_file, root_image_dir):
