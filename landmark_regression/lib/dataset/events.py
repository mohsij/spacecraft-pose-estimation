# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class EventsDataset(JointsDataset):
    def __init__(self, cfg, root, image_dir, image_set, is_train, transform=None, numpy_transform=None):
        super().__init__(cfg, root, image_set, is_train, transform, numpy_transform, multi_scale_target=False)

        self.DATA_DIR = image_dir

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.flip_pairs = []
        self.parent_ids = []

        self.upper_body_ids = None
        self.lower_body_ids = None

        self.image_width = cfg.DATASET.IMAGE_WIDTH
        self.image_height = cfg.DATASET.IMAGE_HEIGHT
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, self.image_set+'.json'
        )

        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        image_annots = {image_anno["id"] : image_anno for image_anno in anno["images"]}
        for a in anno["annotations"]:
            image_name = image_annots[a['image_id']]['file_name']

            box = np.array(a['bbox']).flatten()
            c,s=self._box2cs(box)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)

            joints_reshaped = np.array(a['keypoints']).reshape((-1,3))
            joints_3d[:, 0:2] = joints_reshaped[:, 0:2]
            # Convert detectron visibility labels to mpii.
            joints_3d_vis[:, 0] = joints_reshaped[:, -1] - 1
            joints_3d_vis[:, 1] = joints_reshaped[:, -1] - 1

            image_dir = self.DATA_DIR

            x, y, w, h = box[:4]

            gt_db.append(
                {
                    'image': os.path.join(image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'box_w': w,
                    'box_h': h,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db


    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        # if w > self.aspect_ratio * h:
        #     h = w * 1.0 / self.aspect_ratio
        # elif w < self.aspect_ratio * h:
        #     w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.5  ############### vis ############################### vis ################

        return center, scale


    def evaluate(self, cfg, preds, output_dir, pred_file_name, *args, **kwargs):

        # # convert 0-based index to 1-based index
        # preds[:, :, 0:2] += 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, '{}.mat'.format(pred_file_name))
            savemat(pred_file, mdict={'preds': preds})

        return {'Null': 0}, 0

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
