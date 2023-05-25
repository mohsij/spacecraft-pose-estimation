import argparse
import os
import subprocess
from pathlib import Path

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the pose estimation pipeline with the given trained object detection model and landmark regression model.')
    parser.add_argument(
        "--data_dir", required=True, type=str, 
        help="directory with testing event aedat files")
    parser.add_argument(
        "--detection_model_file", required=True, type=str, 
        help="path to trained object detection model file relative to object detection dir")
    parser.add_argument(
        "--regression_model_file", required=True, type=str, 
        help="path to trained landmark regression model file relative to hrnet dir")
    parser.add_argument(
        "--detection_annotations_base", required=True, type=str, 
        help="directory to export detection annotations to")
    parser.add_argument(
        "--regression_annotations_base", required=True, type=str, 
        help="directory to export landmark regression annotations to")
    parser.add_argument(
        "--pose_estimation_base", required=True, type=str, 
        help="directory to export pose estimation results to")
    parser.add_argument(
        "--validation_annotations", required=True, type=str, 
        help="path to detection validation set annotations file")
    parser.add_argument(
        "--landmarks_file", required=True, type=str, 
        help="blender landmarks file")
    parser.add_argument(
        "--calibration_file_path", required=True, type=str, 
        help="Path to the calibration file")
    parser.add_argument(
        "--image_width", type=int, default=640, help="Image width")
    parser.add_argument(
        "--image_height", type=int, default=480, help="Image height")
    parser.add_argument(
        "--joints_count", type=int, default=24, help="Number of landmarks")
    
    args = parser.parse_args()

    os.chdir("object_detection")

    # Do object detection
    scenes_dir = os.path.join("..", args.data_dir)
    for data_name in os.listdir(scenes_dir):
        if os.path.isdir(os.path.join(scenes_dir, data_name)):
            subprocess.run(["python", "export_object_detection_bounding_boxes.py", 
                            "--frames_dir", os.path.join(scenes_dir, data_name, 'event-frames'), 
                            "--model_file", args.detection_model_file, 
                            "--validation_annotations", args.validation_annotations, 
                            "--landmarks_file", args.landmarks_file, 
                            "--output_dir", os.path.join(args.detection_annotations_base, data_name), 
                            "--image_width", str(args.image_width), "--image_height", str(args.image_height)])

    os.chdir("../landmark_regression")

    makedir(args.regression_annotations_base)
    # Do landmark regression
    scenes_dir = os.path.join("..", "..", args.data_dir)
    for data_name in os.listdir(scenes_dir):
        if os.path.isdir(os.path.join(scenes_dir, data_name)):
            subprocess.run(["python", "tools/test.py", 
                            "--cfg", "experiments/events/events-config.yaml", 
                            "DATA_DIR", os.path.join(scenes_dir, data_name, 'event-frames'), 
                            "OUTPUT_DIR", os.path.join(args.regression_annotations_base, data_name), 
                            "DATASET.ROOT", os.path.join("../../object_detection", args.detection_annotations_base, data_name), 
                            "DATASET.TEST_SET", "test", 
                            "DATASET.TRAIN_SET", "synthetic_train", 
                            "DATASET.IMAGE_WIDTH", str(args.image_width), 
                            "DATASET.IMAGE_HEIGHT", str(args.image_height), 
                            "MODEL.NUM_JOINTS", str(args.joints_count), 
                            "TEST.MODEL_FILE", args.regression_model_file])

    os.chdir("../../pose_estimation")
    scenes_dir = os.path.join("..", args.data_dir)
    for data_name in os.listdir(scenes_dir):
        if os.path.isdir(os.path.join(scenes_dir, data_name)):
            subprocess.run(["python", "export_predicted_poses_real.py", 
                            "--frames_dir", os.path.join(scenes_dir, data_name, 'event-frames'), 
                            "--detection_annotations", os.path.join("../object_detection", args.detection_annotations_base, data_name, "test.json"), 
                            "--pose_annotations", os.path.join("../landmark_regression", args.regression_annotations_base, data_name, "EventsDataset/pose_hrnet/events-config/pred.mat"), 
                            "--landmarks_file", args.landmarks_file,
                            "--calibration_file_path", os.path.join('..', args.calibration_file_path),
                            "--output_dir", os.path.join(args.pose_estimation_base, data_name)])

if __name__ == "__main__":
    main()