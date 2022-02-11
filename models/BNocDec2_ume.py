import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.net_utils import (
    vg_crop,
    hmgt,
    get_scale,
    kabsch_trans_rot,
    rotation_matrix_to_angle_axis,
    angle_axis_to_rotation_matrix,
    get_aligned2noc,
)
from utils.net_utils import cats
from utils.umeyama import umeyama_torch
from typing import Optional, List

from trimesh.transformations import rotation_matrix, translation_matrix, scale_matrix

from dvis import dvis


class BNocDec2_ume(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net
        self.bbox_shape = np.array(conf['bbox_shape'])
        self.gt_augm = conf['gt_augm']
        self.crit = nn.L1Loss().cuda()
        self.noc_weight = conf['weights']['noc']
        self.rot_weight = conf['weights']['rot']
        self.transl_weight = conf['weights']['transl']
        self.scale_weight = conf['weights']['scale']
        self.noc_samples = conf['noc_samples']

    def forward(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bbbox_lvl0: List):

        bx_d0_crops = []
        for B in range(len(bbbox_lvl0)):
            bbox_lvl0 = bbbox_lvl0[B]
            bbox_lvl2 = (bbox_lvl0 / 4).int()

            x_de2_crops = vg_crop(torch.cat([x_d2[B : B + 1], x_e2[B : B + 1]], 1), bbox_lvl2)
            x_e1_crops = vg_crop(x_e1[B : B + 1], bbox_lvl2 * 2)
            # simple interpolate
            x_de2_batch_crops = torch.cat(
                [
                    F.interpolate(x_de2_crop, size=self.bbox_shape.tolist(), mode="trilinear", align_corners=True)
                    for x_de2_crop in x_de2_crops
                ]
            )
            x_e1_batch_crops = torch.cat(
                [
                    F.interpolate(x_e1_crop, size=(self.bbox_shape * 2).tolist(), mode="trilinear", align_corners=True)
                    for x_e1_crop in x_e1_crops
                ]
            )

            x_d0_batch_crops = self.net.forward(x_de2_batch_crops, x_e1_batch_crops)
            x_d0_crops = [
                F.interpolate(
                    x_d0_batch_crops[i : i + 1],
                    size=tuple(bbox_lvl0[i, 3:6] - bbox_lvl0[i, :3]),
                    mode="trilinear",
                    align_corners=True,
                )
                for i in range(len(x_d0_batch_crops))
            ]

            bx_d0_crops.append(x_d0_crops)
        return bx_d0_crops

    def infer_step(
        self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, rpn_gt, bbbox_lvl0: List, binst_occ=None,
    ):

        bx_d0_crops = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)

        bpred_aligned2scans = []
        for B in range(len(bbbox_lvl0)):
            pred_aligned2scans = []

            bbox_lvl0 = bbbox_lvl0[B]
            best_rot_angles_y = []
            for j in range(len(bbox_lvl0)):
                bbox = bbox_lvl0[j]
                inst_occ = binst_occ[B][j]

                scan_noc_inst_crops_grid_coords = torch.nonzero(inst_occ).float()
                pred_noc_on_gt_inst = bx_d0_crops[B][j][0, :, inst_occ].T
                try:
                    pred_noc2scan_t, pred_noc2scan_c, pred_noc2scan_R = self.nocs_to_tcr(
                        pred_noc_on_gt_inst, scan_noc_inst_crops_grid_coords + bbox[:3]
                    )

                    pred_noc2scan = torch.eye(4)
                    pred_noc2scan[:3, :3] = pred_noc2scan_c * pred_noc2scan_R
                    pred_noc2scan[:3, 3] = pred_noc2scan_t

                    pred_aligned2scan = pred_noc2scan @ get_aligned2noc()
                except:
                    pred_noc2scan = torch.eye(4)
                    pred_noc2scan[:3, 3] = (bbox[3:6] + bbox[:3]) / 2
                    pred_aligned2scan = pred_noc2scan @ get_aligned2noc()

                pred_aligned2scans.append(pred_aligned2scan)
            bpred_aligned2scans.append(pred_aligned2scans)

        return {"noc_values": bx_d0_crops, "pred_aligned2scans": bpred_aligned2scans}



    def nocs_to_tr(self, nocs, coords):
        # returns nocs2coords transform
        indices = np.random.choice(range(len(nocs)), min(len(nocs), self.noc_samples))
        t, R = kabsch_trans_rot(nocs[indices].T, coords[indices].T)
        # +
        return t, R

    def nocs_to_tcr(self, nocs, coords):

        indices = np.random.choice(range(len(nocs)), min(len(nocs), self.noc_samples))
        R, c, t = umeyama_torch(nocs[indices], coords[indices])
        # +
        return t, c, R

