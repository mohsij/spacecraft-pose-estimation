import cv2
import random
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.structures import Instances
from object_detection_utils import *
import torch
from torch.utils.data import DataLoader, Dataset
from numpy import ndarray

from pathlib import Path
from typing import Iterable, List, NamedTuple

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class ImageDataset(Dataset):

    def __init__(self, images: List[Path], test_path: Path):
        self.test_set_path = test_path
        self.images = images

    def __getitem__(self, index) -> ndarray:
        return cv2.imread(os.path.join(self.test_set_path, self.images[index]))

    def __len__(self):
        return len(self.images)

class MultiPredictor(DefaultPredictor):
    def __init__(self, cfg, batch_size, workers, test_path):
        super().__init__(cfg)
        self.batch_size = batch_size
        self.workers = workers
        self.test_set_path = test_path

    def __call__(self, images: List[Path]) -> Iterable[Instances]:
        """[summary]

        :param images: [description]
        :type images: List[Path]
        :yield: Predictions for each image
        :rtype: [type]
        """
        dataset = ImageDataset(images, self.test_set_path)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                results: List[Instances] = self.model(batch)
                yield from [self.__map_predictions(result['instances']) for result in results]

    def __collate(self, batch):
        data = []
        for image in batch:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __map_predictions(self, instances: Instances):
        boxes = instances.get('pred_boxes').to("cpu").tensor.detach().numpy()
        scores = instances.get('scores').to("cpu").detach().numpy()

        return [boxes, scores]


def get_iou(bb1, bb2):
    """
    From: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : list
        [x1, y2, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_dist_to_centre(box, centre):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    w = x2 - x1
    h = y2 - y1
    box_centre = [x1 + w/2.0, y1 + h/2.0]
    return np.linalg.norm(np.array(box_centre) - np.array(centre))

def draw_boxes(img, boxes, scores):
    colors = [(0, 255, 0), (160, 112, 255)]
    for i, box in enumerate(boxes):
        x = int(box[0])
        y = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        positions = [(x, y + 30),(x, y + 60)]
        rect_positions = [(x, y), (x, y + 30)]
        cv2.rectangle(img, (x,y), (x2, y2), colors[i], 2)
        # Plot the score.
        cv2.rectangle(img, rect_positions[i], (rect_positions[i][0] + 200, rect_positions[i][1] + 40), (50, 50, 50), -1)
        cv2.putText(img, 'score: %.2f' % scores[i] , positions[i], cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2, 2)

def calc_area(box):
    x1 = 0
    y1 = 1
    x2 = 2
    y2 = 3
    return abs(box[x1] - box[x2]) * abs(box[y1] - box[y2])

def area_of_intersection(box1, box2):
    ''' Length of intersecting part i.e 
        start from max(l1[x], l2[x]) of 
        x-coordinate and end at min(r1[x],
        r2[x]) x-coordinate by subtracting 
        start from end we get required 
        lengths 
        Adapted from : https://www.geeksforgeeks.org/total-area-two-overlapping-rectangles/'''
    x1 = 0
    y1 = 1
    x2 = 2
    y2 = 3
    x_dist = (min(box1[x2], box2[x2]) -
              max(box1[x1], box2[x1]))
 
    y_dist = (min(box1[y2], box2[y2]) -
              max(box1[y1], box2[y1]))

    if x_dist > 0 and y_dist > 0:
        return x_dist * y_dist
    else:
        return 0
 
skeleton = []

info_dict = {
    "description": "SEENIC Object Detection",
    "url": "https://idklol",
    "version": "1.0",
    "year": 2022,
    "contributor": "Australian Institute of Machine Learning",
    "date_created": "2022"
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
        description='Evaluate object detection.')
    parser.add_argument(
        "--frames_dir", required=True, type=str, 
        help="directory with testing event frames")
    parser.add_argument(
        "--model_file", required=True, type=str, 
        help="path to trained object detection model file")
    parser.add_argument(
        "--validation_annotations", required=True, type=str, 
        help="path to validation set annotations file")
    parser.add_argument(
        "--landmarks_file", required=True, type=str, 
        help="blender landmarks file")
    parser.add_argument(
        "--output_dir", required=True, type=str, 
        help="output directory")
    parser.add_argument(
        "--config", required=False, type=str, default="config_4", 
        help="Name of the config in object_detection_utils e.g config_4")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Testing batch size")
    parser.add_argument(
        "--image_width", type=int, default=1280, help="Image width")
    parser.add_argument(
        "--image_height", type=int, default=720, help="Image height")

    args = parser.parse_args()

    with open(args.validation_annotations, 'r') as json_file:
        detectron_dict_validation = json.load(json_file)


    def my_validation_dataset():
        return detectron_dict_validation


    DatasetCatalog.register("my_dataset_validation", my_validation_dataset)
    MetadataCatalog.get("my_dataset_validation").thing_classes = ["spacecraft"]
    MetadataCatalog.get("my_dataset_validation").thing_colors = [(173, 216, 230)]

    cfg = eval(args.config)(args.image_height)

    landmarks_count = pd.read_csv(
        args.landmarks_file)[['x', 'y', 'z']].values.shape[0]

    categories_dicts[0]["keypoints"] = ['p' + str(count + 1) for count in range(landmarks_count)]

    cfg.MODEL.WEIGHTS = args.model_file
    cfg.TEST.DETECTIONS_PER_IMAGE = 2
    test_metadata = MetadataCatalog.get("my_dataset_validation")

    test_set_path = args.frames_dir

    def find_unique_directory_name(output_base_dir):
        suffix = 0
        while os.path.exists(output_base_dir + '_' + str(suffix)):
            suffix += 1
        if suffix == 0:
            return output_base_dir
        else:
            return output_base_dir + '_' + str(suffix)


    output_base_dir = find_unique_directory_name(args.output_dir)
    
    output_dir0 = os.path.join(output_base_dir, 'bounding_box_0')
    output_dir1 = os.path.join(output_base_dir, 'bounding_box_1')
    output_dir2 = os.path.join(output_base_dir, 'bounding_box_2')
    Path(output_dir0).mkdir(parents=True, exist_ok=True)
    Path(output_dir1).mkdir(parents=True, exist_ok=True)
    Path(output_dir2).mkdir(parents=True, exist_ok=True)

    files = os.listdir(test_set_path)
    files = sorted(files)

    predictor = MultiPredictor(cfg, args.batch_size, 1, test_set_path)

    outputs = predictor(files)

    images_dicts = []
    annotations_dicts = []
    for image_read_path, output in tqdm(zip(files, outputs)):
        boxes = output[0]
        scores = output[1]

        full_img_path = os.path.join(test_set_path, image_read_path)
        im = cv2.imread(full_img_path)

        output_dir = output_dir0
        output_box = None
        is_outputting_annotation = True
        if len(boxes) == 2:
            output_dir = output_dir2
        elif len(boxes) == 1:
            output_dir = output_dir1
        else:
            output_dir = output_dir0
            boxes = np.array([[0,0,args.image_width,args.image_height]])
            scores = np.array([0])

        output_box = boxes[scores.argmax()].tolist()
        output_score = scores.max()

        x = output_box[0]
        y = output_box[1]
        w = output_box[2] - output_box[0]
        h = output_box[3] - output_box[1]
        bounding_box = [x, y, w, h]

        image_save_path = os.path.join(output_dir, image_read_path)
        draw_boxes(im, boxes, scores)
        cv2.imwrite(image_save_path, im)

        if is_outputting_annotation:

            image_id = int(os.path.basename(full_img_path).replace('img', '').split('.')[0])
            images_dict = {"license": 1,
                           "file_name": image_read_path,
                           "width": args.image_width,
                           "height": args.image_height,
                           "date_captured": "2022",
                           "id": image_id}

            images_dicts.append(images_dict)

            annotations_dict = {"segmentation": {},
                                "num_keypoints": landmarks_count,
                                "area": w * h,
                                "iscrowd": 0,
                                "keypoints": np.full((landmarks_count * 3), 2.0).tolist(),
                                "image_id": image_id,
                                "bbox": bounding_box,
                                "category_id": 1,
                                "id": image_id
                                }

            annotations_dicts.append(annotations_dict)

    coco_dict = {"info": info_dict,
                 "licenses": licenses_dicts,
                 "categories": categories_dicts,
                 "images": images_dicts,
                 "annotations": annotations_dicts}
    with open(os.path.join(output_base_dir, 'real_test.json'), 'w') as detectron_file:
        detectron_file.write(json.dumps(coco_dict, indent=2))

if __name__ == "__main__":
    main()