import cv2
import json
import math
import time
import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from PIL import Image

from kornia.geometry.conversions import angle_axis_to_quaternion
from kornia.geometry.conversions import QuaternionCoeffOrder

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def cv_rotation_matrix_to_quat(r):
    # Adapted from
    # Theory of Applied Robotics: Kinematics, Dynamics,
    # and Control, Reza N. Jazar, 2nd Edition, Springer, 2010, p. 110, eq. (3.149)-(3.152).

    mat_1 = 0
    mat_2 = 1
    mat_3 = 2

    e0 = math.sqrt(1 + r[mat_1][mat_1] + r[mat_2][mat_2] + r[mat_3][mat_3]) / 2.0
    e1 = math.sqrt(1 + r[mat_1][mat_1] - r[mat_2][mat_2] - r[mat_3][mat_3]) / 2.0
    e2 = math.sqrt(1 - r[mat_1][mat_1] + r[mat_2][mat_2] - r[mat_3][mat_3]) / 2.0
    e3 = math.sqrt(1 - r[mat_1][mat_1] - r[mat_2][mat_2] + r[mat_3][mat_3]) / 2.0

    max_index = np.array([e0, e1, e2, e3]).argmax()
    if max_index == 0:
        e1 = (r[mat_3][mat_2] - r[mat_2][mat_3]) / (4 * e0)
        e2 = (r[mat_1][mat_3] - r[mat_3][mat_1]) / (4 * e0)
        e3 = (r[mat_2][mat_1] - r[mat_1][mat_2]) / (4 * e0)
    elif max_index == 1:
        e2 = (r[mat_2][mat_1] + r[mat_1][mat_2]) / (4 * e1)
        e3 = (r[mat_3][mat_1] + r[mat_1][mat_3]) / (4 * e1)
        # :NOTE:
        # Correction had to be made here because the corresponding equation in the book gave an incorrect result.
        # Particularly, we subtract instead of adding here.
        e0 = (r[mat_3][mat_2] - r[mat_2][mat_3]) / (4 * e1)
    elif max_index == 2:
        e3 = (r[mat_3][mat_2] + r[mat_2][mat_3]) / (4 * e2)
        e0 = (r[mat_1][mat_3] - r[mat_3][mat_1]) / (4 * e2)
        e1 = (r[mat_2][mat_1] + r[mat_1][mat_2]) / (4 * e2)
    else:
        e0 = (r[mat_2][mat_1] - r[mat_1][mat_2]) / (4 * e3)
        e1 = (r[mat_3][mat_1] + r[mat_1][mat_3]) / (4 * e3)
        e2 = (r[mat_3][mat_2] + r[mat_2][mat_3]) / (4 * e3)

    return np.array([e0, e1, e2, e3])

def plot_points(img, xs, ys):
    for x, y in zip(xs, ys):
        img = cv2.circle(img, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
    return img

def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q / np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project(q, r, K, points = np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    p_axes = np.column_stack((points, np.ones(points.shape[0])))
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    x0, y0 = (points_camera_frame[0], points_camera_frame[1])

    # apply distortion
    dist = [0,0,0,0,0]

    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
    x1 = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
    y1 = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0

    # projection to image plane
    x = K[0, 0] * x1 + K[0, 2]
    y = K[1, 1] * y1 + K[1, 2]

    return np.column_stack((x, y))


def main():
    parser = argparse.ArgumentParser(
        description='event frames to pose estimation results.')
    parser.add_argument(
        "--frames_dir", required=True, type=str, 
        help="directory with blender event frames")
    parser.add_argument(
        "--detection_annotations", required=True, type=str, 
        help="file with object detection results in COCO format")
    parser.add_argument(
        "--pose_annotations", required=True, type=str, 
        help="file with pose estimation results")
    parser.add_argument(
        "--landmarks_file", required=True, type=str, 
        help="blender landmarks file")
    parser.add_argument(
        "--calibration_file_path", required=True, type=str, 
        help="file with camera calibration parameters")
    parser.add_argument(
        "--output_dir", required=True, type=str, 
        help="output directory")

    args = parser.parse_args()

    poses = []

    frames_dir = args.frames_dir
    output_dir = args.output_dir

    makedir(output_dir)

    blender_landmarks = pd.read_csv(args.landmarks_file)[['x', 'y', 'z']].values

    with open(args.calibration_file_path, 'r') as calibration_file:
        calibration_parameters = json.load(calibration_file)

    assert(os.path.exists(frames_dir), "Frames directory does not exist")
    assert(os.path.exists(output_dir), "Output directory already exists")

    with open(args.detection_annotations, 'r') as annotations_file:
        annotation_dicts = json.load(annotations_file)

    image_id_list = [annotation['id'] for annotation in annotation_dicts["images"]]
    image_name_dict = {annotation['id']:annotation['file_name'] for annotation in annotation_dicts["images"]}


    preds = scio.loadmat(args.pose_annotations)
    preds = np.array(preds['preds'])
    correspondences = [{'image_id': image_id, 'keypoints': keypoints.flatten()} for image_id, keypoints in
                       zip(image_id_list, preds)]

    for i, correspondence in tqdm(enumerate(correspondences)):
        image_id = correspondence['image_id']
        
        image_points = np.array(correspondence['keypoints']).reshape((-1,3))[:,:2].astype(np.float32)

        # DVX calibration
        K = np.array(calibration_parameters["intrinsics"]["camera_matrix"])
        distortion_coefficients = np.array(calibration_parameters["intrinsics"]["distortion_coefficients"])

        # filter based on confidence.
        confidence_values = np.array(correspondence['keypoints']).reshape((-1,3))[:,-1].astype(np.float32)
        confidence_threshold = 0.95
        good_confidence_indices = confidence_values > confidence_threshold
        max_iters = 100
        curr_iters = 0
        while np.sum(good_confidence_indices) < 15:
            confidence_threshold *= 0.8
            good_confidence_indices = confidence_values > confidence_threshold
            curr_iters += 1
            if(curr_iters >= max_iters):
                break

        ret, pred_rotation_vector, pred_T, inliers = cv2.solvePnPRansac(
            blender_landmarks[good_confidence_indices], image_points[good_confidence_indices], K,
            distCoeffs=distortion_coefficients, flags=cv2.SOLVEPNP_EPNP, iterationsCount=10000, reprojectionError=15.0)

        pred_rotation_matrix, _ = cv2.Rodrigues(pred_rotation_vector)
        pred_rotation_in_quat = cv_rotation_matrix_to_quat(pred_rotation_matrix)

        rt = np.column_stack((pred_rotation_matrix, pred_T))

        krt = K @ rt

        stacked = np.column_stack((blender_landmarks, np.ones(blender_landmarks.shape[0])))

        points_calc = krt @ np.transpose(stacked)
        points_calc /= points_calc[2]
        points_calc = np.transpose(points_calc)

        file_name_read = image_name_dict[image_id]
        file_name_write = os.path.basename(file_name_read).split('.')[0] + '.jpg'
        img = cv2.imread(os.path.join(frames_dir, file_name_read))
        #undistorted_img = cv2.undistort(img, K, distortion_coefficients)
        undistorted_img = img

        xa = points_calc[:, 0]
        ya = points_calc[:, 1]
        poses.append({"image_name": file_name_read,
                      "T": pred_T.tolist(),
                      "rotation_matrix": pred_rotation_matrix.tolist()})

        bbox = annotation_dicts['annotations'][i]['bbox']
        bbox = [int(coordinate) for coordinate in bbox]
        cv2.rectangle(undistorted_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,255,0), 2)

        undistorted_img = plot_points(undistorted_img, xa, ya)
        cv2.imwrite(os.path.join(args.output_dir, file_name_write), undistorted_img)

    with open(os.path.join(output_dir, 'opencv_poses.json'), 'w') as json_file:
        json_file.write(json.dumps(poses, indent=2))

if __name__ == "__main__":
    main()