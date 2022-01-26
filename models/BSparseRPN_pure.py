import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.net_utils import vg_crop
from utils.anchor_handler import AnchorHandler
from typing import Optional, List
from torch_cluster import fps, nearest
from block_timer.timer import Timer
#from vision3d.layers import nms3d

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
        
    
    def validation_step(self, x_e2: torch.Tensor, bdscan):
        bpred_reg_values = self.forward(x_e2)

        losses = self.loss(bpred_reg_values, bdscan)

        bpred_bboxes, bgt_target, brpn_conf = self._to_bboxes(bpred_reg_values, bdscan, self.conf['min_conf_test'], self.conf['max_proposals_test'])
        # bdscan.bbboxes ~ bpred_bboxes
       
        return (bpred_bboxes, bgt_target, brpn_conf), losses, {}, {}
    
    def infer_step(self, x_e2: torch.Tensor, bdscan):
        bpred_reg_values = self.forward(x_e2)

        bpred_bboxes, bgt_target, brpn_conf = self._to_bboxes(bpred_reg_values, bdscan, self.conf['min_conf_test'], self.conf['max_proposals_test'])
        # bdscan.bbboxes ~ bpred_bboxes
       
        return (bpred_bboxes, bgt_target, brpn_conf)
    
    def _to_bboxes(self, bpred_reg_values, bdscan, min_conf, max_proposals):
        # bdscan only used to get gt_targets
        bpred_bboxes = []
        brpn_conf = []
        bgt_target = []

        pred_coords_l, pred_reg_values_l = bpred_reg_values.decomposed_coordinates_and_features
        # dvis(bdscan.breg_sparse.C[bdscan.breg_sparse.F[:,0]>0,1:] + bdscan.breg_sparse.F[bdscan.breg_sparse.F[:,0]>0,1:4].cpu())
        for B,(pred_coords, pred_reg_values) in enumerate(zip(pred_coords_l, pred_reg_values_l)):
            
            pred_logits = pred_reg_values[:,0:2]
            pred_conf = F.softmax(pred_logits, 1)[:, 1]
            conf_mask = pred_conf >= min(min_conf, pred_conf.max())

            pred_conf = pred_conf[conf_mask]
            pred_reg_values = pred_reg_values[conf_mask]
            pred_delta_t = pred_reg_values[:, 2:5].contiguous()
            pred_delta_s = pred_reg_values[:, 5:9].contiguous()

            pred_center = pred_delta_t + pred_coords[conf_mask].cuda()

            fps_batch = torch.ones(len(pred_center)).long().cuda()
            fps_indices = fps(pred_center, fps_batch, min(0.99, max_proposals / len(pred_center)))
            # merge fps points if spatially close
            pairwise_dist_fps = torch.cdist(pred_center[fps_indices].unsqueeze(0), pred_center[fps_indices].unsqueeze(0))[0]
            
            
            pairwise_close = pairwise_dist_fps < np.sqrt(200)
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
            #pred_bboxes[3:6] = torch.min(pred_bboxes[3:6], torch.Tensor([192, 96, 192]).cuda(), dim=1)
            #pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3],)
            pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3], max=bdscan.bscan_shape[B][0]-1)
            pred_bboxes[:, 4] = torch.clamp(pred_bboxes[:, 4], max=bdscan.bscan_shape[B][1]-1)
            pred_bboxes[:, 5] = torch.clamp(pred_bboxes[:, 5], max=bdscan.bscan_shape[B][2]-1)

            # easy lazy: TODO IMPROVE CLUSTERING (here later centroids collide)
            nms_filtered = nms3d(pred_bboxes.float(), pred_confs, 0.4)
            pred_bboxes = pred_bboxes[nms_filtered]
            pred_confs = pred_confs[nms_filtered]

            min_size_filter = torch.all((pred_bboxes[:, 3:6] - pred_bboxes[:,:3]) >= 5, 1)
            
            #if min_size_filter.sum() == 0:
            #    min_size_filter[0] = True


            pred_bboxes = pred_bboxes[min_size_filter]
            pred_confs = pred_confs[min_size_filter]

            # ensure max proposals
            if len(pred_bboxes) > max_proposals:
                pred_bboxes = pred_bboxes[:max_proposals]
                pred_confs = pred_confs[:max_proposals]


            bpred_bboxes.append(pred_bboxes)

            brpn_conf.append(pred_conf)

            # find matching gt
            if len(bdscan.bbboxes) > 0:
                overlaps = bbox_overlaps(pred_bboxes.float(), bdscan.bbboxes[B].float())
                if len(overlaps) >0:
                    gt_target = [bdscan.bobj_inds[B][i] for i in torch.argmax(overlaps, 1)]
                else:
                    #TODO IS THIS CORRECT???
                    gt_target = []
            else:
                gt_target = len(pred_bboxes)*[0]
            
            bgt_target.append(gt_target)


           
        return bpred_bboxes, bgt_target, brpn_conf
        

    def training_step(self, x_e2: torch.Tensor, bdscan):
        #with Timer('forward'):
        bpred_reg_values = self.forward(x_e2)
        #with Timer('loss'):
        losses = self.loss(bpred_reg_values, bdscan)
        #with Timer('to bboxes'):
        bpred_bboxes, bgt_target, brpn_conf = self._to_bboxes(bpred_reg_values, bdscan, self.conf['min_conf_train'], self.conf['max_proposals_train'])
        # bdscan.bbboxes ~ bpred_bboxes
        return (bpred_bboxes, bgt_target, brpn_conf), losses, {}, {}


    def loss(self, bpred_reg_values: torch.Tensor,  bdscan):
        
        bobjness_loss = []
        bbbox_loss = []
        bfp_loss = []
        bweighted_loss = []

        # per batch
        # check speed if better parallel
        pred_coords_l, pred_reg_values_l = bpred_reg_values.decomposed_coordinates_and_features
        gt_coords_l, gt_reg_values_l = bdscan.breg_sparse.decomposed_coordinates_and_features
        for (pred_coords, pred_reg_values, gt_coords, gt_reg_values) in zip(pred_coords_l, pred_reg_values_l, gt_coords_l, gt_reg_values_l):
            #if pred_coords != gt_coords:
            #    print("DAFIUQ")
             
            pos_mask = gt_reg_values[:, 0] > 0
            neg_mask = gt_reg_values[:, 0] == 0

            objness_loss = F.cross_entropy(pred_reg_values[:,:2], pos_mask.long()).unsqueeze(0)

            if (pos_mask).sum() == 0:
                import pdb
                print(bdscan.bseq_name)
                print(bdscan.bscan_idx)
                pdb.set_trace()
                

            bbox_loss = F.smooth_l1_loss(pred_reg_values[pos_mask,2:]/10, gt_reg_values[pos_mask,1:]/10).unsqueeze(0)

            bobjness_loss.append(objness_loss)

            bbbox_loss.append(bbox_loss)

            weighted_loss = self.objness_weight * objness_loss + self.bbox_weight * bbox_loss
            bweighted_loss.append(weighted_loss)




           
        losses = {'bweighted_loss': bweighted_loss, 'bobjness_loss': bobjness_loss, 'bbbox_loss': bbbox_loss}
        return losses
        