""" Collection of functions to transform popular datasets into torch_dataset Datasets """
import os
import tqdm

import json
from scipy.io import loadmat

from .detection_dataset import DetectionDataset
from .siamese_dataset import SiameseDataset


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

        if no_crowd and ann['iscrowd'] == 1:
            continue

        import warnings
        warnings.filterwarnings('error')
        try:
            dataset.set_ann(
                image_id=ann['image_id'],
                bbox=bbox,
                class_name=orig_label_to_name[ann['category_id']],
                segmentation=ann['segmentation']
            )
        except:
            continue

    return dataset


def convert_wider_to_detection_dataset(
    wider_mat_file,
    root_image_dir,
    allowed_blur_labels=[0,1,2],
    allowed_expression_labels=[0,1],
    allowed_illumination_labels=[0,1],
    allowed_occlusion_labels=[0,1,2],
    allowed_pose_labels=[0,1]
):
    """ Converts a wider annotation file to a detection dataset (which can be saved with save_dataset)
    Args
        wider_mat_file  : The annotation file eg. 'XXX/wider_face_val.mat'
        root_image_dir : The folder storing all image folders, 'The directory with 0--Prade ...'
    Returns
        DetectionDataset object containing wider data
    """
    # Load wider data
    wider_data = loadmat(wider_mat_file)

    # Create empty dataset object
    dataset = DetectionDataset(root_dir=root_image_dir)

    # Set classes even though there are only 1 class
    dataset.set_classes(['person_face'])

    # Set image and annotations one image at a time
    # Loop through events
    for event_id, event in enumerate(wider_data['event_list']):
        event_name = str(event[0][0])
        event_files  = wider_data['file_list'][event_id,0]

        event_bboxes              = wider_data['face_bbx_list'][event_id,0]
        event_blur_labels         = wider_data['blur_label_list'][event_id,0]
        event_expression_labels   = wider_data['expression_label_list'][event_id,0]
        event_illumination_labels = wider_data['illumination_label_list'][event_id,0]
        event_occlusion_labels    = wider_data['occlusion_label_list'][event_id,0]
        event_pose_labels         = wider_data['pose_label_list'][event_id,0]
        event_invalid_labels      = wider_data['invalid_label_list'][event_id,0]

        pbar = tqdm.tqdm(total=len(event_files), desc='Setting {}'.format(event_name))

        # Loop through each image
        for file_id, file_name in enumerate(event_files):
            pbar.update(1)
            # Save image
            file_name = str(file_name[0][0])
            image_info = dataset.set_image(image_path=os.path.join(event_name, file_name) + '.jpg')
            image_id = image_info['id']

            bboxes              = event_bboxes[file_id,0]
            blur_labels         = event_blur_labels[file_id,0]
            expression_labels   = event_expression_labels[file_id,0]
            illumination_labels = event_illumination_labels[file_id,0]
            occlusion_labels    = event_occlusion_labels[file_id,0]
            pose_labels         = event_pose_labels[file_id,0]
            invalid_labels      = event_invalid_labels[file_id,0]

            # Loop though each annotation
            for i in range(len(bboxes)):
                # do checks
                if not (
                    (blur_labels[i,0] in allowed_blur_labels) and
                    (expression_labels[i,0] in allowed_expression_labels) and
                    (illumination_labels[i,0] in allowed_illumination_labels) and
                    (occlusion_labels[i,0] in allowed_occlusion_labels) and
                    (pose_labels[i,0] in allowed_pose_labels) and
                    (invalid_labels[i,0] == 0)
                ):
                    continue

                bbox = bboxes[i].copy().tolist()
                bbox[3] += bbox[1]
                bbox[2] += bbox[0]

                import warnings
                warnings.filterwarnings('error')
                try:
                    dataset.set_ann(
                        image_id=image_id,
                        bbox=bbox,
                        class_name='person_face'
                    )
                except:
                    continue

        pbar.close()

    return dataset


def convert_lfw_to_siamese_dataset(people_txt_file, root_image_dir):
    """ Converts a lfw people file to a siamese dataset (which can be saved with save_dataset)
    Args
        people_txt_file : The annotation file eg. 'XXX/peopleDevTrain.txt'
        root_image_dir  : The folder storing all image folders, 'The directory with Aaron_Eckhart ...'
    Returns
        SiameseDataset object containing lfw data
    """
    # Load lfw data
    with open(people_txt_file, 'r') as f:
        lfw_data = f.readlines()[1:]

    # Create empty dataset object
    dataset = SiameseDataset(root_dir=root_image_dir)

    # Set images one person at a time
    for person_info in tqdm.tqdm(lfw_data, desc='Setting each person'):
        person_name = person_info.split('\t')[0]
        person_folder = os.path.join(root_image_dir, person_name)

        for image_file in os.listdir(person_folder):
            dataset.set_image(
                image_path=os.path.join(person_name, image_file),
                class_name=person_name
            )

    return dataset
