import cv2
import random
import json
import argparse
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from object_detection_utils import *

def main():
    parser = argparse.ArgumentParser(
        description='Train object detection.')
    parser.add_argument(
        "--train_annotations", required=True, type=str, 
        help="path to training set annotations file")
    parser.add_argument(
        "--validation_annotations", required=True, type=str, 
        help="path to validation set annotations file")
    parser.add_argument(
        "--train_images_dir", required=True, type=str, 
        help="path to images")
    parser.add_argument(
        "--validation_images_dir", required=True, type=str, 
        help="path to images")
    parser.add_argument(
        "--output_dir", required=True, type=str, 
        help="output directory")
    parser.add_argument(
        "--config", required=False, type=str, default="config_4", 
        help="Name of the config in object_detection_utils e.g config_4")
    parser.add_argument(
        "--image_width", type=int, default=1280, help="Image width")
    parser.add_argument(
        "--image_height", type=int, default=720, help="Image height")

    args = parser.parse_args()

    setup_logger()

    register_coco_instances("my_dataset_train", {}, args.train_annotations, args.train_images_dir)
    register_coco_instances("my_dataset_validation", {}, args.validation_annotations, args.validation_images_dir)

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, config, dataset_name, output_folder=None):
            if output_folder is None:
                os.makedirs("detection_eval", exist_ok=True)
                output_folder = "detection_eval"
            return COCOEvaluator(dataset_name, config, False, output_folder, max_dets_per_image=1)


    cfg = eval(args.config)(args.image_height)

    cfg.OUTPUT_DIR = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    
if __name__ == "__main__":
    main()