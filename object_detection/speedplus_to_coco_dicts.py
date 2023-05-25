import numpy as np
import json
import os
import scipy.io
import argparse
from pathlib import Path

from speed_plus_utils.utils import Camera
from speed_plus_utils.utils import project

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_visible_keypoints(points, width, height):
    visible_points = []
    for point in points:
        if 0 < point[0] < width and point[1] > 0 and point[1] < height:
            # Visibility should be 2 if the keypoint is visible 
            # and labelled.
            visible_points.append(np.array([point[0], point[1], 2]))
        else:
            # Visibility should be 1 if the keypoint is visible but 
            # not labelled.
            # :NOTE:
            # For speedplus we just treat all points as visible though
            visible_points.append(np.array([point[0], point[1], 2]))
    return np.array(visible_points)

skeleton = []

info_dict = {
    "description": "Dataset in COCO Format",
    "url": "myurl",
    "version": "1.0",
    "year": 2021,
    "contributor": "Your Name",
    "date_created": "2021"
}
licenses_dicts = [{
    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License"
}]
categories_dicts = [{
    "supercategory": "spacecraft",
    "id": 1,
    "name": "spacecraft",
    "keypoints": [],
    "skeleton": skeleton
}]

# The pose file is a json list with each item having a the fileName, the quaternion rotation and translation labels.
file_name_key = 'filename'
rotation_key = 'q_vbs2tango_true'
translation_key = 'r_Vo2To_vbs_true'

def main():
    parser = argparse.ArgumentParser(
        description='Convert SPEED+ to COCO format.')
    parser.add_argument(
        "--dataset_dir", type=str, default="../datasets/speedplus",
        help="path to the directory containing SPEED+ data e.g ../dataset/speedplus")
    parser.add_argument(
        "--landmarks_mat_file", type=str, default="speed_plus_utils/pts11.mat",
        help="path to pts11.mat for SPEED+")
    parser.add_argument(
        "--dataset_type", type=str, default="synthetic" ,
        help="dataset type one of: [synthetic,lightbox,sunlamp]")
    parser.add_argument(
        "--dataset_split", type=str, default="train" ,
        help="split of the dataset corresponding to a json file in the SPEED+ set i.e one of [train, validation, test]")
    parser.add_argument(
        "--output_dir", type=str, default="speedplus_dicts",
        help="output directory for the COCO json file")

    args = parser.parse_args()

    dataset_type = args.dataset_type
    split = args.dataset_split

    # We are testing the speed plus dataset here.
    dataset_path = os.path.join(args.dataset_dir, dataset_type)

    # Load Bo's 3D landmark points.
    landmark_points_MAT = scipy.io.loadmat(args.landmarks_mat_file)

    # Landmark points are in pixels, so convert them to m to be consistent in units.
    landmark_points = np.array(landmark_points_MAT['pts']) * Camera.ppx

    with open(os.path.join(dataset_path, split + '.json'), 'r') as pose_json:
        poses = json.load(pose_json)

    # Get all images
    image_indices = range(0, len(poses))

    images_dicts = []
    annotations_dicts = []

    for i, image_index in enumerate(image_indices):
        translationVector = poses[image_index][translation_key]
        rotation_in_quat = poses[image_index][rotation_key]
        points_2d = project(rotation_in_quat, translationVector, landmark_points)
        image_path = poses[image_index]['filename']
        full_img_path = os.path.join(dataset_path, 'images', image_path)

        # Get the bounding box coordinates.
        min_x = points_2d[:, 0].min()
        min_y = points_2d[:, 1].min()
        max_x = points_2d[:, 0].max()
        max_y = points_2d[:, 1].max()
        box_origin_x = min_x
        box_origin_y = min_y
        box_width = max_x - min_x
        box_height = max_y - min_y

        # Expand the bounding box by 10% just so everything fits.
        x_tolerance = box_width * 0.1
        y_tolerance = box_width * 0.1
        box_origin_x -= x_tolerance
        box_origin_y -= y_tolerance
        box_width += 2 * x_tolerance
        box_height += 2 * y_tolerance

        # Write as detectron instance dictionary.
        bounding_box = [box_origin_x, box_origin_y, box_width, box_height]

        image_id = image_path.split('.')[0][3:].lstrip('0')
        images_dict = {"license": 1,
                       "file_name": image_path,
                       "width": 1920,
                       "height": 1200,
                       "date_captured": "2021",
                       "id": int(image_id)}
        images_dicts.append(images_dict)

        visible_keypoints = get_visible_keypoints(points_2d, 1900, 1200).flatten()

        annotations_dict = {"segmentation": {},
                            "num_keypoints": 11,
                            "area": box_width * box_height,
                            "iscrowd": 0,
                            "keypoints": visible_keypoints.tolist(),
                            "image_id": int(image_id),
                            "bbox": bounding_box,
                            "category_id": 1,
                            "id": i
                            }

        annotations_dicts.append(annotations_dict)

    coco_dict = {"info": info_dict,
                 "licenses": licenses_dicts,
                 "categories": categories_dicts,
                 "images": images_dicts,
                 "annotations": annotations_dicts}

    makedir(args.output_dir)
    with open(os.path.join(args.output_dir, dataset_type + "_" + split + '.json'), 'w') as coco_file:
        coco_file.write(json.dumps(coco_dict, indent=2))

if __name__ == "__main__":
    main()
