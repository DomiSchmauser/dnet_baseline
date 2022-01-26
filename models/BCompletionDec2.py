import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.net_utils import vg_crop
from typing import Optional, List

from utils.net_utils import cats
from dvis import dvis

class BCompletionDec2(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net
        self.bbox_shape = np.array(conf['bbox_shape'])
        self.gt_augm = conf['gt_augm']
        self.w  = conf['total_weight']
        self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2]).cuda()).cuda()

    def forward(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bbbox_lvl0: List):
        bx_d0_crops = []
        for B in range(len(bbbox_lvl0)):
            bbox_lvl0 = bbbox_lvl0[B]
            bbox_lvl2 = (bbox_lvl0 / 4).int()

            x_de2_crops = vg_crop(torch.cat([x_d2[B:B+1], x_e2[B:B+1]], 1), bbox_lvl2)
            x_e1_crops = vg_crop(x_e1[B:B+1], bbox_lvl2*2)
            # simple interpolate
            x_de2_batch_crops = torch.cat([F.interpolate(x_de2_crop, size=self.bbox_shape.tolist(), mode='trilinear', align_corners=True) for x_de2_crop in x_de2_crops])
            x_e1_batch_crops = torch.cat([F.interpolate(x_e1_crop, size=(self.bbox_shape*2).tolist(), mode='trilinear', align_corners=True) for x_e1_crop in x_e1_crops])

            x_d0_batch_crops = self.net.forward(x_de2_batch_crops, x_e1_batch_crops)
            x_d0_crops = [F.interpolate(x_d0_batch_crops[i:i + 1], size=tuple(bbox_lvl0[i, 3:6] - bbox_lvl0[i,:3]), mode='trilinear', align_corners=True) for i in range(len(x_d0_batch_crops))]
            
            bx_d0_crops.append(x_d0_crops)
        return bx_d0_crops
        
    
    def validation_step(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bdscan, bbbox_lvl0: List, bgt_target: List):
        if not all([len(bbox_lvl0)>0 for bbox_lvl0 in bbbox_lvl0]):
            return [[]], {'bweighted_loss': [torch.Tensor([0.0]).cuda()], 'bcompl_loss': [torch.Tensor([0.0]).cuda()]}, {}, {}
        bx_d0_crops = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)
        losses = self.loss(bx_d0_crops, bbbox_lvl0, bgt_target, bdscan)
        pred_compl = [[x_d0_crop > 0.0 for x_d0_crop in x_d0_crops] for x_d0_crops in bx_d0_crops]
        return pred_compl, losses, {}, {}
    
    def infer_step(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bbbox_lvl0: List):
        if not all([len(bbox_lvl0)>0 for bbox_lvl0 in bbbox_lvl0]):
            return [[]]
        bx_d0_crops = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)
        pred_compl = [[x_d0_crop > 0.0 for x_d0_crop in x_d0_crops] for x_d0_crops in bx_d0_crops]
        return pred_compl

    def training_step(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bdscan, bbbox_lvl0: List, bgt_target: List):
        if self.gt_augm:
            bgt_bbox, bgt_obj_inds = bdscan.bbboxes, bdscan.bobj_inds
            bbbox_lvl0 =  [cats(bbox_lvl0, gt_bbox,0)  for bbox_lvl0, gt_bbox in zip(bbbox_lvl0, bgt_bbox)]   
            bgt_target =  [gt_target + gt_obj_inds  for gt_target, gt_obj_inds in zip(bgt_target, bgt_obj_inds)]
        
        bx_d0_crops = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)

        losses = self.loss(bx_d0_crops, bbbox_lvl0, bgt_target, bdscan)

        return bx_d0_crops, losses, {}, {}

    

    def loss(self, bx_d0_crops, bbbox_lvl0: List, bgt_target: List, bdscan):
        bcompl_loss = []
        bweighted_loss = []
        for B in range(len(bbbox_lvl0)):
            scan_inst_mask_crops = vg_crop(bdscan.bscan_inst_mask[B:B+1], bbbox_lvl0[B])
            scan_compl_crops = [(scan_inst_mask_crop == int(bgt_target[B][i])).float() for i,scan_inst_mask_crop in enumerate(scan_inst_mask_crops)]
            
            # debugging
            if False:
                i= 0
                dvis(bx_d0_crops[B][i][0, 0] > 0.5, c=2)
                dvis(scan_compl_crops[i][0,0], add=True, c=3)
                

            compl_loss = torch.stack([self.crit(x_d0_crop, scan_compl_crop) for x_d0_crop, scan_compl_crop in zip(bx_d0_crops[B], scan_compl_crops)])
            weighted_loss = self.w * compl_loss
            bcompl_loss.append(compl_loss)
            bweighted_loss.append(weighted_loss)
        
        losses = {'bweighted_loss': bweighted_loss, 'bcompl_loss': bcompl_loss }
        return losses
