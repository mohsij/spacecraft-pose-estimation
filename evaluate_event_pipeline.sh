#!/bin/bash

ARGUMENT_LIST=(
  "data_dir"
  "detection_model_file"
  "regression_model_file"
  "detection_annotations_base"
  "regression_annotations_base"
  "pose_estimation_base"
  "validation_annotations"
  "landmarks_file"
  "image_width"
  "image_height"
  "joints_count"
  "calibration_file_path"
)


# read arguments
opts=$(getopt \
  --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
  --name "$(basename "$0")" \
  --options "" \
  -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      data_dir=$2
      shift 2
      ;;

    --detection_model_file)
      detection_model_file=$2
      shift 2
      ;;

    --regression_model_file)
      regression_model_file=$2
      shift 2
      ;;

    --detection_annotations_base)
      detection_annotations_base=$2
      shift 2
      ;;

    --regression_annotations_base)
      regression_annotations_base=$2
      shift 2
      ;;

    --pose_estimation_base)
      pose_estimation_base=$2
      shift 2
      ;;

    --validation_annotations)
      validation_annotations=$2
      shift 2
      ;;

    --landmarks_file)
      landmarks_file=$2
      shift 2
      ;;

    --image_width)
      image_width=$2
      shift 2
      ;;

    --image_height)
      image_height=$2
      shift 2
      ;;

    --joints_count)
      joints_count=$2
      shift 2
      ;;

    --calibration_file_path)
      calibration_file_path=$2
      shift 2
      ;;

    *)
      break
      ;;
  esac
done

source $CONDA_PREFIX/etc/profile.d/conda.sh &&
conda activate v2e &&

cd v2e &&
pwd &&
up_level="../" &&
python convert_aedats.py --scenes_dir "$up_level$data_dir" --calibration_file_path "$up_level$calibration_file_path" --image_width $image_width --image_height $image_height &&

conda deactivate &&
conda activate tdrs-pose &&

cd .. &&

python evaluate_pipeline.py --data_dir $data_dir \
                            --detection_model_file $detection_model_file \
                            --regression_model_file $regression_model_file \
                            --detection_annotations_base $detection_annotations_base --regression_annotations_base $regression_annotations_base \
                            --pose_estimation_base $pose_estimation_base --validation_annotations $validation_annotations \
                            --landmarks_file $landmarks_file --image_width $image_width --image_height $image_height --joints_count $joints_count \
                            --calibration_file_path $calibration_file_path
