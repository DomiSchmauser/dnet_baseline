import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.net_utils import vg_crop
from utils.anchor_handler import AnchorHandler
from typing import Optional, List
from torch_cluster import fps, nearest
from block_timer.timer import Timer
from vision3d.layers import nms3d

from dvis import dvis

from utils.net_utils import iou3d

from utils.anchor_handler import bbox_overlaps

class BSparseRPN_pure(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net
        self.conf = conf

        self.objness_weight = 1.0
        self.bbox_weight = 1.0


    def forward(self, x_e2: torch.Tensor):
        bpred_reg_values = self.net.forward(x_e2)
        return bpred_reg_values
        
    
    def validation_step(self, x_e2: torch.Tensor, rpn_gt):
        bpred_reg_values = self.forward(x_e2)

        losses = self.loss(bpred_reg_values, rpn_gt)

        bpred_bboxes, bgt_target, brpn_conf = self._to_bboxes(bpred_reg_values, rpn_gt, self.conf['min_conf_test'], self.conf['max_proposals_test'])
        # bdscan.bbboxes ~ bpred_bboxes
       
        return (bpred_bboxes, bgt_target, brpn_conf), losses, {}, {}
    
    def infer_step(self, x_e2: torch.Tensor, rpn_gt):
        bpred_reg_values = self.forward(x_e2)

        bpred_bboxes, bgt_target, brpn_conf = self._to_bboxes(bpred_reg_values, rpn_gt, self.conf['min_conf_test'], self.conf['max_proposals_test'])
        # bdscan.bbboxes ~ bpred_bboxes
       
        return (bpred_bboxes, bgt_target, brpn_conf)
    
    def _to_bboxes(self, bpred_reg_values, rpn_gt, min_conf, max_proposals):
        # bdscan only used to get gt_targets
        bpred_bboxes = []
        brpn_conf = []
        bgt_target = []

        pred_coords_l, pred_reg_values_l = bpred_reg_values.decomposed_coordinates_and_features
        for B,(pred_coords, pred_reg_values) in enumerate(zip(pred_coords_l, pred_reg_values_l)):

            # Objectness score
            pred_logits = pred_reg_values[:,0:2]
            pred_conf = F.softmax(pred_logits, 1)[:, 1]
            conf_mask = pred_conf >= min(min_conf, pred_conf.max())

            pred_conf = pred_conf[conf_mask]
            pred_reg_values = pred_reg_values[conf_mask]
            pred_delta_t = pred_reg_values[:, 2:5].contiguous() # translational distance obj center to voxel
            pred_delta_s = pred_reg_values[:, 5:9].contiguous() # size

            pred_center = pred_delta_t + pred_coords[conf_mask].cuda()

            fps_batch = torch.ones(len(pred_center)).long().cuda()
            fps_indices = fps(pred_center, fps_batch, min(0.99, max_proposals / len(pred_center)))
            # merge fps points if spatially close
            pairwise_dist_fps = torch.cdist(pred_center[fps_indices].unsqueeze(0), pred_center[fps_indices].unsqueeze(0))[0]

            pairwise_close = pairwise_dist_fps < np.sqrt(200) #
            centroid_clusters = []
            joint_ids = []
            
            for i in range(len(fps_indices)):
                if i not in joint_ids:
                    # new centroid_cluster
                    centroid_cluster = []
                    for j in range(i, len(fps_indices)):
                        if pairwise_close[i, j]:
                            centroid_cluster.append(fps_indices[j])
                            joint_ids.append(j)
                    centroid_clusters.append(centroid_cluster)

            centroid_centers =  torch.stack([ pred_center[[centroid_cluster]].mean(0) for centroid_cluster in centroid_clusters ])

            pairwise_dist = torch.cdist(centroid_centers, pred_center.unsqueeze(0))[0]
            clusters = pairwise_dist < np.sqrt(200)

            pred_bbox_sizes = torch.stack([torch.mean(pred_delta_s[clusters[i]], 0) for i in range(len(clusters))])
            pred_bbox_centers = torch.stack([torch.mean(pred_center[clusters[i]], 0) for i in range(len(clusters))])

            pred_confs = torch.stack([torch.mean(pred_conf[clusters[i]], 0) for i in range(len(clusters))])

            pred_bboxes = torch.clamp(torch.cat([pred_bbox_centers - pred_bbox_sizes, pred_bbox_centers + pred_bbox_sizes], 1).round().int(), 0)
            pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3], max=rpn_gt['scan_shape'][0]-1) # x max
            pred_bboxes[:, 4] = torch.clamp(pred_bboxes[:, 4], max=rpn_gt['scan_shape'][1]-1) # y max
            pred_bboxes[:, 5] = torch.clamp(pred_bboxes[:, 5], max=rpn_gt['scan_shape'][2]-1) # z max in voxel coords

            nms_filtered = nms3d(pred_bboxes.float(), pred_confs, 0.4)
            pred_bboxes = pred_bboxes[nms_filtered]
            pred_confs = pred_confs[nms_filtered]

            min_size_filter = torch.all((pred_bboxes[:, 3:6] - pred_bboxes[:,:3]) >= 5, 1) # extents larger 5 in discrete space

            # Breaks at size predicted boxes
            pred_bboxes = pred_bboxes[min_size_filter]
            pred_confs = pred_confs[min_size_filter]

            # ensure max proposals
            if len(pred_bboxes) > max_proposals:
                pred_bboxes = pred_bboxes[:max_proposals]
                pred_confs = pred_confs[:max_proposals]

            bpred_bboxes.append(pred_bboxes)

            brpn_conf.append(pred_conf)

            # find matching gt
            if len(rpn_gt['bboxes'][B]) > 0: # List with all bboxes in the scan
                overlaps = bbox_overlaps(pred_bboxes.float(), rpn_gt['bboxes'][B].float())
                if len(overlaps) >0:
                    gt_target = [rpn_gt['bobj_idxs'][B][i] for i in torch.argmax(overlaps, 1)]
                else:
                    gt_target = []
            else:
                gt_target = len(pred_bboxes)*[0]
            
            bgt_target.append(gt_target)


        return bpred_bboxes, bgt_target, brpn_conf
        

    def training_step(self, x_e2: torch.Tensor, rpn_gt):

        bpred_reg_values = self.forward(x_e2) # num occ x 8 featuredim
        losses = self.loss(bpred_reg_values, rpn_gt)
        bpred_bboxes, bgt_target, brpn_conf = self._to_bboxes(bpred_reg_values, rpn_gt, self.conf['min_conf_train'], self.conf['max_proposals_train'])

        return (bpred_bboxes, bgt_target, brpn_conf), losses, {}, {}


    def loss(self, bpred_reg_values: torch.Tensor,  rpn_gt):
        
        bobjness_loss = []
        bbbox_loss = []
        bweighted_loss = []

        # per batch
        pred_coords_l, pred_reg_values_l = bpred_reg_values.decomposed_coordinates_and_features # in sparse tensor format list N x 3, N x 8
        gt_coords_l, gt_reg_values_l = rpn_gt['breg_sparse'].decomposed_coordinates_and_features # gt occupancies N x 7
        for (pred_coords, pred_reg_values, gt_coords, gt_reg_values) in zip(pred_coords_l, pred_reg_values_l, gt_coords_l, gt_reg_values_l):

            pos_mask = gt_reg_values[:, 0] > 0 # calculated w > 0 # occupied
            #neg_mask = gt_reg_values[:, 0] == 0

            objness_loss = F.cross_entropy(pred_reg_values[:,:2], pos_mask.long()).unsqueeze(0)

            if (pos_mask).sum() == 0:
                import pdb
                pdb.set_trace()

            # center coord, extent x,y,z similar to votenet, target = delta t and delta s
            bbox_loss = F.smooth_l1_loss(pred_reg_values[pos_mask,2:]/10, gt_reg_values[pos_mask,1:]/10).unsqueeze(0) # removed division by 10

            bobjness_loss.append(objness_loss)

            bbbox_loss.append(bbox_loss)

            weighted_loss = self.objness_weight * objness_loss + self.bbox_weight * bbox_loss
            bweighted_loss.append(weighted_loss)


        losses = {'bweighted_loss': bweighted_loss, 'bobjness_loss': bobjness_loss, 'bbbox_loss': bbbox_loss}
        return losses
        