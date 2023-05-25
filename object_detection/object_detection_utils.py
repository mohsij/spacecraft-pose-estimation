import os
from detectron2.config import get_cfg
from detectron2 import model_zoo


def config_base(config_file, train_set, val_set, weights_file):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = (val_set,)

    cfg.DATALOADER.NUM_WORKERS = 4

    # Let training initialize from model zoo.
    cfg.MODEL.WEIGHTS = weights_file

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    return cfg

checkpoint_dir = os.path.join('models', 'model_zoo_checkpoints')


def config_1():
    cfg = config_base("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                      "my_dataset_train",
                      "my_dataset_validation",
                      os.path.join(checkpoint_dir, 'faster_rcnn_X_101_32x8d_FPN_3x_checkpoint.pkl'))

    # Adjust up if val mAP is still rising, adjust down if overfit.
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.STEPS = (500, 800, 1000, 1100, 1200)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.TEST.EVAL_PERIOD = 1000
    return cfg


def config_2():
    cfg = config_base("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                      "my_dataset_train",
                      "my_dataset_validation",
                      os.path.join(checkpoint_dir, 'faster_rcnn_R_101_FPN_3x_checkpoint.pkl'))

    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = (2000, 6000, 8000)

    cfg.TEST.EVAL_PERIOD = 2000

    return cfg

def config_3():
    cfg = config_base("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                      "my_dataset_train",
                      "my_dataset_validation",
                      os.path.join(checkpoint_dir, 'faster_rcnn_X_101_32x8d_FPN_3x_checkpoint.pkl'))

    cfg.OUTPUT_DIR = "output_sunlamp"

    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.IMS_PER_BATCH = 7
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.STEPS = (10000,)

    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    #cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True

    cfg.INPUT.MIN_SIZE_TRAIN = (1200,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1200
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT.MIN_SIZE_TEST = 1200
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 1200

    cfg.INPUT.RANDOM_FLIP = "none"

    cfg.TEST.EVAL_PERIOD = 5000

    return cfg

def config_4(image_height):
    cfg = config_base("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                      "my_dataset_train",
                      "my_dataset_validation",
                      os.path.join(checkpoint_dir, 'faster_rcnn_X_101_32x8d_FPN_3x_checkpoint.pkl'))

    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.STEPS = (8000,)

    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    #cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True

    cfg.INPUT.MIN_SIZE_TRAIN = (image_height,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = image_height
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT.MIN_SIZE_TEST = image_height
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = image_height

    cfg.INPUT.RANDOM_FLIP = "none"

    cfg.TEST.EVAL_PERIOD = 5000

    return cfg    

def config_20():
    cfg = config_base("COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                      "my_dataset_train",
                      "my_dataset_validation",
                      os.path.join(checkpoint_dir, 'retinanet_R_101_FPN_3x_checkpoint.pkl'))

    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.STEPS = (6000, 10000, 15000)

    # cfg.INPUT.MIN_SIZE_TRAIN = (1200,)
    # # Sample size of smallest side by choice or random selection from range give by
    # # INPUT.MIN_SIZE_TRAIN
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # # Maximum size of the side of the image during training
    # cfg.INPUT.MAX_SIZE_TRAIN = 1920
    # # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    # cfg.INPUT.MIN_SIZE_TEST = 1200
    # # Maximum size of the side of the image during testing
    # cfg.INPUT.MAX_SIZE_TEST = 1920

    cfg.TEST.EVAL_PERIOD = 10000

    return cfg


