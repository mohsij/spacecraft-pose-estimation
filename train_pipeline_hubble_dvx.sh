#!/bin/bash

source $CONDA_PREFIX/etc/profile.d/conda.sh &&
conda activate v2e &&

cd synthetic-data/hubble-dvx

v2e -i frames --overwrite --input_frame_rate=100 \
--timestamp_resolution=.01 --disable_slomo --auto_timestamp_resolution=False \
--dvs_exposure duration 0.2 --output_folder=output_0.2 --overwrite --pos_thres=.15 --neg_thres=.15 \
--sigma_thres=0.3 --dvs_text events.csv --output_width=640 --output_height=480 --cutoff_hz=30 --avi_frame_rate=10

v2e -i frames --overwrite --input_frame_rate=100 \
--timestamp_resolution=.01 --disable_slomo --auto_timestamp_resolution=False \
--dvs_exposure duration 0.1 --output_folder=output_0.1 --overwrite --pos_thres=.15 --neg_thres=.15 \
--sigma_thres=0.3 --dvs_text events.csv --output_width=640 --output_height=480 --cutoff_hz=30 --avi_frame_rate=10


v2e -i frames --overwrite --input_frame_rate=100 \
--timestamp_resolution=.01 --disable_slomo --auto_timestamp_resolution=False \
--dvs_exposure duration 0.05 --output_folder=output_0.05 --overwrite --pos_thres=.15 --neg_thres=.15 \
--sigma_thres=0.3 --dvs_text events.csv --output_width=640 --output_height=480 --cutoff_hz=30 --avi_frame_rate=10


v2e -i frames --overwrite --input_frame_rate=100 \
--timestamp_resolution=.01 --disable_slomo --auto_timestamp_resolution=False \
--dvs_exposure duration 0.01 --output_folder=output_0.01 --overwrite --pos_thres=.15 --neg_thres=.15 \
--sigma_thres=0.3 --dvs_text events.csv --output_width=640 --output_height=480 --cutoff_hz=30 --avi_frame_rate=10

cd ../../object_detection

conda deactivate &&
conda activate tdrs-pose &&

python split_images.py --frames_dir ../synthetic-data/hubble-merged/output/event-frames

python events_to_coco_dicts.py --frames_dir ../synthetic-data/hubble-merged/output/event-frames \
                               --gt_dir ../synthetic-data/hubble-merged/ground_truth \
                               --landmarks_file ../pose_estimation/projection_utils/landmarks_hubble.csv \
                               --output_prefix synthetic --output_dir hubble_synthetic_annotations_multi_exp_dvx --image_width 640 --image_height 480  &&

python train_object_detection.py --train_annotations hubble_synthetic_annotations_multi_exp_dvx/synthetic_train.json \
                                 --validation_annotations hubble_synthetic_annotations_multi_exp_dvx/synthetic_validation.json \
                                 --image_dir ../synthetic-data/hubble-merged/output/event-frames \
                                 --output_dir hubble_synthetic_model_multi_exp_dvx_aug --image_width=640 --image_height=480 &&

#train landmark regression

cd ../landmark_regression
python tools/train.py --cfg experiments/events/events-config.yaml DATA_DIR ../../synthetic-data/hubble-merged/output/event-frames \
                      OUTPUT_DIR hubble_synthetic_model_multi_exp_dvx_aug DATASET.ROOT ../../object_detection/hubble_synthetic_annotations_multi_exp_dvx \
                      DATASET.TEST_SET synthetic_validation DATASET.TRAIN_SET synthetic_train \
                      DATASET.IMAGE_WIDTH 640 DATASET.IMAGE_HEIGHT 480 MODEL.NUM_JOINTS 24
