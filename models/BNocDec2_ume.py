import numpy as np
import torch
from torch import nn
import traceback
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
from utils.umeyama import umeyama_torch, estimateSimilarityUmeyama
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

            x_de2_crops = vg_crop(torch.cat([x_d2[B:B + 1], x_e2[B:B + 1]], 1), bbox_lvl2)
            x_e1_crops = vg_crop(x_e1[B:B + 1], bbox_lvl2 * 2)
            # simple interpolate
            x_de2_batch_crops = torch.cat(
                [F.interpolate(x_de2_crop, size=self.bbox_shape.tolist(), mode='trilinear', align_corners=True) for
                 x_de2_crop in x_de2_crops])
            x_e1_batch_crops = torch.cat(
                [F.interpolate(x_e1_crop, size=(self.bbox_shape * 2).tolist(), mode='trilinear', align_corners=True) for
                 x_e1_crop in x_e1_crops])

            x_d0_batch_crops = self.net.forward(x_de2_batch_crops, x_e1_batch_crops)
            x_d0_crops = [F.interpolate(x_d0_batch_crops[i:i + 1], size=tuple(
                (bbox_lvl0[i, 3:6] - bbox_lvl0[i, :3]).detach().cpu().to(torch.int).tolist()),
                                        mode='trilinear', align_corners=True) for i in range(len(x_d0_batch_crops))]

            bx_d0_crops.append(x_d0_crops)
        return bx_d0_crops

    def infer_step(
        self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, rpn_gt, bgt_target, bscan_obj, bbbox_lvl0: List, binst_occ=None,
    ):

        bx_d0_crops = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)
        brot_errors = []
        btransl_errors = []
        bpred_aligned2scans = []
        for B in range(len(bbbox_lvl0)):
            pred_aligned2scans = []
            rot_errors = []
            transl_errors = []

            gt_target = bgt_target[B]
            scan_obj = bscan_obj[B]

            bbox_lvl0 = bbbox_lvl0[B]
            for j in range(len(bbox_lvl0)):
                bbox = bbox_lvl0[j]
                inst_occ = binst_occ[B][j]

                scan_noc_inst_crops_grid_coords = torch.nonzero(inst_occ).float()
                pred_noc_on_gt_inst = bx_d0_crops[B][j][0, :, inst_occ].T

                # GT
                aligned2noc = scan_obj[str(gt_target[j])]['aligned2noc']
                noc2scan = scan_obj[str(gt_target[j])]['noc2scan']

                # using GT scale
                scaled_pred_nocs = pred_noc_on_gt_inst * get_scale(noc2scan)[0] #with gt scale works

                try:
                    pred_noc2scan_t, pred_noc2scan_R = self.nocs_to_tr( #noc2scan
                        scaled_pred_nocs, scan_noc_inst_crops_grid_coords + bbox[:3]  # pred_noc_on_gt_inst
                    )
                except:
                    print('also issue')
                    traceback.print_exc()

                s = torch.diag(get_scale(noc2scan)[:3]).to(torch.float32)
                pred_noc2scan = torch.eye(4)
                pred_noc2scan[:3, :3] = pred_noc2scan_R @ s
                pred_noc2scan[:3, 3] = pred_noc2scan_t
                pred_aligned2scan = pred_noc2scan @ aligned2noc

                # Pose predictions
                gt_scaled_noc2scan_R = (noc2scan[:3, :3] / get_scale(noc2scan[:3, :3])).to(
                    torch.float32)
                relative_compl_loss_R = pred_noc2scan_R @ torch.inverse(gt_scaled_noc2scan_R) # Compare unscaled rotations
                delta_rot = rotation_matrix_to_angle_axis(relative_compl_loss_R.unsqueeze(0))[0]
                rot_error = torch.abs(delta_rot) * 180 / np.pi
                if not torch.isnan(rot_error).any():
                    rot_errors.append(torch.unsqueeze(rot_error.detach().cpu(), dim=0))

                transl_error = torch.abs(pred_noc2scan_t.detach().cpu() - noc2scan[:3, 3].detach().cpu()) # compare translation in coord space
                if not torch.isnan(transl_error).any():
                    transl_errors.append(torch.unsqueeze(transl_error, dim=0))

                '''
                try:
                    pred_noc2scan_t, pred_noc2scan_c, pred_noc2scan_R = self.nocs_to_tcr(
                        scaled_pred_nocs, scan_noc_inst_crops_grid_coords + bbox[:3] #Nocs
                    )

                    pred_noc2scan = torch.eye(4)
                    pred_noc2scan[:3, :3] = pred_noc2scan_c * pred_noc2scan_R
                    pred_noc2scan[:3, 3] = pred_noc2scan_t

                    pred_aligned2scan = pred_noc2scan @ aligned2noc

                    # Pose predictions
                    gt_scaled_noc2scan_R = (noc2scan[:3, :3] / get_scale(noc2scan[:3, :3])).to(
                        torch.float32)
                    relative_compl_loss_R = pred_noc2scan_R @ torch.inverse(gt_scaled_noc2scan_R)
                    delta_rot = rotation_matrix_to_angle_axis(relative_compl_loss_R.unsqueeze(0))[0]
                    rot_error = torch.abs(delta_rot) * 180 / np.pi
                    if not torch.isnan(rot_error).any():
                        rot_errors.append(torch.unsqueeze(rot_error.detach().cpu(), dim=0))

                    transl_error = torch.abs(pred_noc2scan_t.detach().cpu() - noc2scan[:3, 3].detach().cpu())
                    if not torch.isnan(transl_error).any():
                        transl_errors.append(torch.unsqueeze(transl_error, dim=0))

                except:
                    print('Nocs exception')
             
                    try:
                        pred_noc2scan_t, pred_noc2scan_R = self.nocs_to_tr(
                            scaled_pred_nocs, scan_noc_inst_crops_grid_coords + bbox[:3] #pred_noc_on_gt_inst
                        )
                    except:
                        print('also issue')
                    
                    traceback.print_exc()
                    # pred_noc2scan[:3, :3] = pred_noc2scan_R
                    # pred_noc2scan[:3, 3] = pred_noc2scan_t

                    pred_noc2scan = torch.eye(4)
                    pred_noc2scan[:3, 3] = (bbox[3:6] + bbox[:3]) / 2
                    pred_aligned2scan = pred_noc2scan @ aligned2noc
                '''

                pred_aligned2scans.append(pred_aligned2scan)
            bpred_aligned2scans.append(pred_aligned2scans)
            brot_errors.append(rot_errors)
            btransl_errors.append(transl_errors)

        return {"noc_values": bx_d0_crops, "pred_aligned2scans": bpred_aligned2scans}, (brot_errors, btransl_errors)



    def nocs_to_tr(self, nocs, coords):
        # returns nocs2coords transform
        indices = np.random.choice(range(len(nocs)), min(len(nocs), self.noc_samples))
        t, R = kabsch_trans_rot(nocs[indices].T, coords[indices].T)
        # +
        return t, R

    def nocs_to_tcr(self, nocs, coords):

        indices = np.random.choice(range(len(nocs)), min(len(nocs), self.noc_samples))
        R, c, t = umeyama_torch(nocs[indices], coords[indices]) # source target
        return t, c, R

