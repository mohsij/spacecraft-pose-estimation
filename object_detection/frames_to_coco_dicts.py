import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import scipy.io
import random
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import detectron2.structures

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
            visible_points.append(np.array([point[0], point[1], 1]))
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

def main():
    parser = argparse.ArgumentParser(
        description='event frames to coco dicts.')
    parser.add_argument(
        "--frames_dir", required=True, type=str, 
        help="directory with blender event frames")
    parser.add_argument(
        "--gt_dir", required=True, type=str, 
        help="directory with blender ground truth files")
    parser.add_argument(
        "--landmarks_file", required=True, type=str, 
        help="blender landmarks file")
    parser.add_argument(
        "--output_prefix", required=True, type=str, default="synthetic",
        help="prefix for output dictionary files")
    parser.add_argument(
        "--output_dir", required=True, type=str, 
        help="output directory")
    parser.add_argument(
        "--image_width", type=int, default=1280, help="Image width")
    parser.add_argument(
        "--image_height", type=int, default=720, help="Image height")

    args = parser.parse_args()

    frames_dir = args.frames_dir
    ground_truth_dir = args.gt_dir

    # Load 3D landmarks.
    blender_landmarks = pd.read_csv(
        args.landmarks_file)[['x', 'y', 'z']].values

    categories_dicts[0]["keypoints"] = ['p' + str(count + 1) for count in range(blender_landmarks.shape[0])]

    splits = ['train', 'test', 'validation']

    for split in splits:
        images_dicts = []
        annotations_dicts = []
        for frame_file_name in tqdm(sorted(os.listdir(os.path.join(frames_dir, split)))):
            frame_count = int(frame_file_name.split('.')[0])
            K = np.loadtxt(
                    os.path.join(
                        ground_truth_dir, 
                        'camera_intrinsics_{}.txt'.format(frame_count)))

            with open(os.path.join(ground_truth_dir, 'pose_{}.json'.format(frame_count))) as json_file:
                pose = json.load(json_file)
                rotation_matrix = np.array(pose['rotation'])
                T = np.array(pose['translation'])

            # Project the landmark points onto the image.
            rt = np.column_stack((rotation_matrix, T))

            krt = K @ rt

            stacked = np.column_stack((blender_landmarks, np.ones(blender_landmarks.shape[0])))

            points_calc = krt @ np.transpose(stacked)
            points_calc /= points_calc[2]
            points_calc = np.transpose(points_calc)
            points_2d = np.column_stack((points_calc[:, 0], points_calc[:, 1]))

            full_img_path = os.path.join(frames_dir, split, frame_file_name)

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

            images_dict = {"license": 1,
                           "file_name": frame_file_name,
                           "width": args.image_width,
                           "height": args.image_height,
                           "date_captured": "2022",
                           "id": int(frame_count)}

            images_dicts.append(images_dict)

            annotations_dict = {"segmentation": {},
                                "num_keypoints": blender_landmarks.shape[0],
                                "area": box_width * box_height,
                                "iscrowd": 0,
                                "keypoints": get_visible_keypoints(points_2d, args.image_width, args.image_height).flatten().tolist(),
                                "image_id": int(frame_count),
                                "bbox": bounding_box,
                                "category_id": 1,
                                "id": int(frame_count)
                                }

            annotations_dicts.append(annotations_dict)

        random.shuffle(images_dicts)
        random.shuffle(annotations_dicts)

        makedir(args.output_dir)
        coco_dict = {"info": info_dict,
                     "licenses": licenses_dicts,
                     "categories": categories_dicts,
                     "images": images_dicts,
                     "annotations": annotations_dicts}
        with open(os.path.join(args.output_dir, args.output_prefix + "_" + split + ".json"), 'w') as detectron_file:
            detectron_file.write(json.dumps(coco_dict, indent=2))

if __name__ == "__main__":
    main()