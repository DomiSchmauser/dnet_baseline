import torch
import numpy as np
import torch.nn.functional as F
from typing import List
#from layer_utils.vision3d.layers import nms3d

def iou3d(boxes, query_boxes):
    box_ares = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])
    query_ares = (
        (query_boxes[:, 3] - query_boxes[:, 0])
        * (query_boxes[:, 4] - query_boxes[:, 1])
        * (query_boxes[:, 5] - query_boxes[:, 2])
    )

    if type(boxes) == torch.Tensor:
        iw = (torch.min(boxes[:, 3], query_boxes[:, 3]) - torch.max(boxes[:, 0], query_boxes[:, 0])).clamp(min=0)
        ih = (torch.min(boxes[:, 4], query_boxes[:, 4]) - torch.max(boxes[:, 1], query_boxes[:, 1])).clamp(min=0)
        il = (torch.min(boxes[:, 5], query_boxes[:, 5]) - torch.max(boxes[:, 2], query_boxes[:, 2])).clamp(min=0)
        ua = (box_ares+ query_ares - iw * ih * il ).float()
        
    else:
        iw = (np.min([boxes[:, 3], query_boxes[:, 3]], axis=0) - np.max([boxes[:, 0], query_boxes[:, 0]], axis=0)).clip(min=0)
        ih = (np.min([boxes[:, 4], query_boxes[:, 4]], axis=0) - np.max([boxes[:, 1], query_boxes[:, 1]], axis=0)).clip(min=0)
        il = (np.min([boxes[:, 5], query_boxes[:, 5]], axis=0) - np.max([boxes[:, 2], query_boxes[:, 2]], axis=0)).clip(min=0)
        ua = (box_ares+ query_ares - iw * ih * il).astype(np.float)
    overlaps = iw * ih * il / ua
    return overlaps


def bbox_overlaps(boxes1, boxes2):
    
    ious = []
    
    for i in range(len(boxes2)):
        box2 = boxes2[i:i+1]
        if type(boxes1) == np.ndarray:
            iou = iou3d(boxes1, np.tile(box2, (len(boxes1),1)))
        else:
            iou = iou3d(boxes1, box2.repeat((len(boxes1),1)))
        ious.append(iou)
    
    if type(boxes1) == np.ndarray:
        return np.stack(ious, 1)
    else:
        return torch.stack(ious,1 )

class AnchorHandler():
    def __init__(self, anchor_shapes, input_shape, feature_stride, rpn_neg, rpn_pos):
        # anchor shapes: A x 3 (whl)
        self._anchor_shapes = np.array(anchor_shapes)
        self._input_shape = input_shape
        self._feature_stride = feature_stride
        self.rpn_neg = rpn_neg
        self.rpn_pos = rpn_pos

        self.anchors = self._generate_anchors()  # A*N x 6 (min, max)

    def _generate_anchors(self):
        # Enumerate shifts in feature space -> N shifts
        shifts_x = np.arange(0, self._input_shape[0], self._feature_stride)
        shifts_y = np.arange(0, self._input_shape[1], self._feature_stride)
        shifts_z = np.arange(0, self._input_shape[2], self._feature_stride)

        
        shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)
        box_w, box_center_x = np.meshgrid(self._anchor_shapes[:, 0], shifts_x)
        box_h, box_center_y = np.meshgrid(self._anchor_shapes[:, 1], shifts_y)
        box_l, box_center_z = np.meshgrid(self._anchor_shapes[:, 2], shifts_z)
        
        box_centers = np.stack([box_center_x, box_center_y, box_center_z], axis=2).reshape([-1, 3])
        box_sizes = np.stack([box_w, box_h, box_l], axis=2).reshape([-1, 3])
        
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5*box_sizes], axis=1)
        #xyzA x 6
        return boxes


    def bboxes2rpn_format(self, bboxes: np.ndarray, obj_inds: List, scan_coverages: np.ndarray):
        # match anchors against bboxes
        overlaps = bbox_overlaps(self.anchors, bboxes)
        anchor_iou_argmax = np.argmax(overlaps, axis=1)

        # assign matching class
        rpn_match = np.zeros([self.anchors.shape[0]], dtype=np.int32)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them.
        rpn_match[(anchor_iou_max < self.rpn_neg)] = -1
         # 2. Set an anchor for each GT box (regardless of IoU value).
        bbox_iou_argmax = np.argmax(overlaps, axis=0)
        rpn_match[bbox_iou_argmax] = 1
        # 3. Set anchors with high overlap as positive.
        rpn_match[anchor_iou_max >= self.rpn_pos] = 1

        
        # generate rpn_bbox
        sizes = bboxes[:, 3:6] - bboxes[:,:3]
        centers = bboxes[:,:3] + 0.5 * sizes

        anchor_sizes = self.anchors[:, 3:6] - self.anchors[:,:3]
        anchor_centers = self.anchors[:,:3] + 0.5 * anchor_sizes

        rpn_bbox = np.concatenate([(centers[anchor_iou_argmax] - anchor_centers) / anchor_sizes,
        np.log(sizes[anchor_iou_argmax] / anchor_sizes)], axis=1).astype(np.float32)

        gt_weights = scan_coverages[anchor_iou_argmax]

        gt_obj_inds = [obj_inds[i] for i in anchor_iou_argmax]
        # WHLA,6 : (dx/Aw,dy/Ah,dz/Al,log(w/Aw), log(h/Ah),log(l/Al))
        return rpn_bbox, rpn_match, gt_weights, gt_obj_inds
    
    def _rpn_bboxes2bboxes(self, rpn_bboxes, anchors):

        anchor_sizes = anchors[:, 3:6] - anchors[:,:3]
        anchor_centers = anchors[:,:3] + 0.5 * anchor_sizes

        centers = rpn_bboxes[:,:3] * anchor_sizes + anchor_centers
        sizes = torch.exp(rpn_bboxes[:, 3:6]) * anchor_sizes

        return torch.cat([centers - sizes / 2, centers + sizes / 2], dim=1)
    
    def rpn_format2bboxes(self, rpn_logits: torch.Tensor, rpn_bboxes: torch.Tensor, gt_obj_inds: List, min_conf: float,  nms_th: float, max_proposals : int):
        # rpn_bboxes WHLA,6 : (dx/Aw,dy/Ah,dz/Al,log(w/Aw), log(h/Ah),log(l/Al))
        rpn_indices = torch.arange(len(rpn_logits))
        rpn_scores = F.softmax(rpn_logits, 1)[:,1]  #foreground scores
        
        scores = rpn_scores

        # ensure min conf
        min_conf_mask = scores >= min(scores.max()-0.01,min_conf)
        scores = scores[min_conf_mask]
        anchors = self.anchors[min_conf_mask]
        rpn_bboxes = rpn_bboxes[min_conf_mask]
        gt_obj_inds = [gt_obj_inds[i] for i in torch.nonzero(min_conf_mask)]
        rpn_indices = rpn_indices[min_conf_mask]
        #scores, order = rpn_scores.sort(descending=True)

      
        bboxes = self._clip_bboxes(self._rpn_bboxes2bboxes(rpn_bboxes ,anchors))

        # ensure bbox big enough
        valid_size_mask = torch.all((bboxes[:, 3:6] - bboxes[:,:3]) > 5, dim=1)
        if valid_size_mask.sum() == 0:
            valid_size_mask[0]= True

        bboxes = bboxes[valid_size_mask]
        scores = scores[valid_size_mask]
        gt_obj_inds = [gt_obj_inds[i] for i in torch.nonzero(valid_size_mask)]
        rpn_indices = rpn_indices[valid_size_mask]
        
        keep = nms3d(bboxes.float(), scores, nms_th)
        # the proposals with the highest score

        bboxes = bboxes[keep]
        scores = scores[keep]
        gt_obj_inds = [gt_obj_inds[i] for i in keep]
        rpn_indices = rpn_indices[keep]

        _, order = scores.sort(descending=True)

        if max_proposals < len(order):
            order = order[:max_proposals]
        
        bboxes = bboxes[order]
        scores = scores[order]
        gt_obj_inds = [gt_obj_inds[i] for i in order]
        rpn_indices = rpn_indices[order]
        """
        bboxes = bboxes[keep]
        gt_obj_inds = [gt_obj_inds[i] for i in keep]
        """
        return bboxes, gt_obj_inds, rpn_indices

    def _clip_bboxes(self, bboxes):
        # bboxes: N,6   (6: min, max in scene grid)
        bboxes = torch.cat([
            torch.clamp(bboxes[:, 0:1], min=0, max=self._input_shape[0]-1),
            torch.clamp(bboxes[:, 1:2], min=0, max=self._input_shape[1]-1),
            torch.clamp(bboxes[:, 2:3], min=0, max=self._input_shape[2]-1),
            torch.clamp(bboxes[:, 3:4], min=0, max=self._input_shape[0]-1),
            torch.clamp(bboxes[:, 4:5], min=0, max=self._input_shape[1]-1),
            torch.clamp(bboxes[:, 5:6], min=0, max=self._input_shape[2]-1)], dim=1).round().int()
        return bboxes

    def filter_with_gt_overlap(self, pred_bboxes, gt_obj_inds, rpn_indices, gt_bboxes):
        pred_gt_iou = iou3d(pred_bboxes.float(), gt_bboxes.float())
        # filter for those with IOU > 0.5
        mask = (pred_gt_iou > 0.01)
        return pred_bboxes[mask], [gt_obj_inds[i] for i,val in enumerate(mask) if val], rpn_indices[mask]





