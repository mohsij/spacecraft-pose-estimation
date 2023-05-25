import subprocess
import os
import argparse
import json
import numpy as np
import cv2
from pathlib import Path

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description='Frames from video files.')
    parser.add_argument(
        "--scenes_dir", required=True, type=str, 
        help="directory to event aedat files")
    parser.add_argument(
        "--calibration_file_path", required=True, type=str, 
        help="Path to the calibration file")
    parser.add_argument(
        "--image_width", type=int, default=640, help="Image width")
    parser.add_argument(
        "--image_height", type=int, default=480, help="Image height")

    args = parser.parse_args()

    calibration_file_path = args.calibration_file_path
    with open(calibration_file_path, 'r') as calibration_file:
        calibration_parameters = json.load(calibration_file)

    camera_matrix = np.array(calibration_parameters["intrinsics"]["camera_matrix"])
    distortion_coefficients = np.array(calibration_parameters["intrinsics"]["distortion_coefficients"])

    base_dir = args.scenes_dir
    for scene in os.listdir(base_dir):
        full_path = os.path.join(base_dir, scene)
        if os.path.isdir(full_path):
            csv_file_name = 'events.csv'
            avi_file_name = 'event-frames-video.avi'
            csv_path = os.path.join(full_path, csv_file_name)
            avi_path = avi_file_name
            # if not os.path.exists(csv_path):
            #     subprocess.run(["python", "aedat_to_csv.py", "--events_file", full_path, "--output_file", csv_path])
            if os.path.exists(csv_path):
                subprocess.run(["python", "e2v.py", "--events_file", csv_path, 
                                "--avi_frame_rate", "30", 
                                "--dvs_vid", avi_path, 
                                "--dvs_exposure", "duration", "10000",
                                "--output_folder", full_path,
                                "--output_width", str(args.image_width), 
                                "--output_height", str(args.image_height)])
                event_frame_distorted_dir = os.path.join(full_path, 'event-frames-distorted')
                event_frame_dir = os.path.join(full_path, 'event-frames')
                makedir(event_frame_dir)
                for image_filename in os.listdir(event_frame_distorted_dir):
                    if 'bmp' in image_filename:
                        img = cv2.imread(os.path.join(event_frame_distorted_dir, image_filename))
                        undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients)
                        cv2.imwrite(os.path.join(event_frame_dir, image_filename), undistorted_img)

if __name__ == "__main__":
    main()