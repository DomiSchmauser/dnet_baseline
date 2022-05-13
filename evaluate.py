import numpy as np
import torch

import logging
import re

from utils.net_utils import iou3d
from pathlib import Path

import pdb

from utils.net_utils import merge_dict, vg_crop, rotation_matrix_to_angle_axis
from typing import Optional, List


from dvis import dvis
import pandas as pd

from utils.eval_utils import mask_iou, l1_acc, mpl_plot

log = logging.getLogger(__name__)

def get_mota(num_gt_objs, num_misses, num_fps, num_switches):
    '''
    Calculates a mota score over all frames seen
    '''
    mota = 1.0 - (float(num_misses + num_fps + num_switches) / float(num_gt_objs))
    return mota


def evaluate(outputs, inputs, losses=None, analyses=None):

    return evaluate_bdscan(outputs, inputs, losses, analyses)

def evaluate_bdscan(outputs, inputs, losses=None, analyses=None):


    # GT data preparation ----------------------------------------------------------------------------------------------

    # Unpack inputs
    dense_features, sparse_features, rpn_features = inputs

    rpn_gt = {}
    rpn_gt['breg_sparse'] = rpn_features[0]
    rpn_gt['scan_shape'] = dense_features.shape[2:]
    rpn_gt['bboxes'] = rpn_features[1]  # list(N boxes x 6)
    rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
    rpn_gt['bscan_inst_mask'] = rpn_features[3]  # list( 1 x X x Y x Z)
    rpn_gt['bscan_nocs_mask'] = rpn_features[4]  # list( 3 x X x Y x Z)
    bscan_obj = rpn_features[5]
    bscan_info = rpn_features[6]
    bcam_info = rpn_features[7]
    bshift_info = rpn_features[8]

    bbbox_lvl0, bgt_target, brpn_conf = outputs['rpn']['bbbox_lvl0'], outputs['rpn']['bgt_target'], outputs['rpn']['brpn_conf']
    
    bdscan_gt_df = pd.DataFrame()
    for B in range(len(rpn_gt['bboxes'])):

        # Tracking infos
        seq_pattern = "test/(.*?)/coco_data"
        scan_pattern = "rgb_(.*?).png"
        seq_name = re.search(seq_pattern, bscan_info[B]).group(1)
        scan_idx = int(re.search(scan_pattern, bscan_info[B]).group(1))
        campose = bcam_info[B]
        shift = bshift_info[B]

        for obj_idx in rpn_gt['bobj_idxs'][B]:

            dscan_gt_df = pd.DataFrame()
            dscan_gt_df['seq_name'] = [seq_name]
            dscan_gt_df['scan_idx'] = [scan_idx]
            dscan_gt_df['obj_idx'] = [obj_idx]
            #dscan_gt_df['model_id'] = [dobj.model_id]
            dscan_gt_df['class_id'] = [bscan_obj[B][str(obj_idx)]['category_id']]
            dscan_gt_df['class_name'] = [bscan_obj[B][str(obj_idx)]['class_name']]
            dscan_gt_df['rot_sym'] = [bscan_obj[B][str(obj_idx)]['rot_sym']]
            dscan_gt_df['aligned2scan'] = [bscan_obj[B][str(obj_idx)]['aligned2scan'].detach().cpu().numpy()]
            dscan_gt_df['scan_coverage'] = [bscan_obj[B][str(obj_idx)]['num_occ']]
            dscan_gt_df['campose'] = [campose]
            dscan_gt_df['shift'] = [shift]

            bdscan_gt_df = pd.concat([bdscan_gt_df, dscan_gt_df], axis=0)
        

    # Predicted data preparation --------------------------------------------------------------------------------------
    bdscan_df = pd.DataFrame()
    for B in range(len(bbbox_lvl0)):
        dscan_df = pd.DataFrame() 
        scan_objs = bscan_obj[B]
        bbox_lvl0, gt_target, rpn_conf = bbbox_lvl0[B], bgt_target[B], brpn_conf[B]

        # Tracking infos
        seq_pattern =  "test/(.*?)/coco_data"
        scan_pattern = "rgb_(.*?).png"
        seq_name = re.search(seq_pattern, bscan_info[B]).group(1)
        scan_idx = int(re.search(scan_pattern, bscan_info[B]).group(1))

        # object level evaluation
        for j, (pred_bbox, obj_idx) in enumerate(zip(bbox_lvl0, gt_target)):
            dobject = scan_objs[str(obj_idx)]
            pred_df = pd.DataFrame()
            info_df = pd.DataFrame()

            info_df['seq_name'] = [seq_name]
            info_df['scan_idx'] = [scan_idx]

            info_df['prop_idx'] = [j]
            info_df['gt_target'] = [obj_idx]
            info_df['pred_aligned2scan'] = [outputs['noc']['pred_aligned2scans'][B][j].numpy()]
            info_df['occ'] = [torch.squeeze(outputs['completion'][B][j]).detach().cpu().numpy()]
            info_df['noc'] = [torch.squeeze(outputs['noc']['noc_values'][B][j]).detach().cpu().numpy()]
            info_df['bbox'] = [outputs['rpn']['bbbox_lvl0'][B][j].detach().cpu().numpy()]

            info_df['class_name'] = [dobject['class_name']]
            info_df['class_id'] = [dobject['category_id']]

            pred_df = pd.concat([pred_df, info_df], axis=1)#, keys=['','info'])

            if 'rpn' in outputs:
                rpn_df = pd.DataFrame()
                gt_bbox = torch.from_numpy(dobject['box_3d']).cuda().int()

                rpn_df['rpn_iou'] = [iou3d(pred_bbox.unsqueeze(0).float(), gt_bbox.unsqueeze(0).float()).item()]
                rpn_df['rpn_conf'] = [rpn_conf[j].item()]

                pred_df = pd.concat([pred_df, rpn_df], axis=1)
            
            # Completion
            compl_df = pd.DataFrame()

            gt_scan_inst_mask_crop = vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox, crop_box=True )
            gt_scan_compl_crop = (gt_scan_inst_mask_crop == int(obj_idx))
            pred_compl_crop = outputs['completion'][B][j]

            gt_unseen_compl_crop = (torch.abs(vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox, crop_box=True )) > 0) & gt_scan_compl_crop
            pred_unseen_compl_crop = (torch.abs(vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox, crop_box=True )) > 0) & pred_compl_crop

            compl_df['compl_iou'] = [mask_iou(pred_compl_crop, gt_scan_compl_crop, 1)]
            compl_df['unseen_iou'] = [mask_iou(pred_unseen_compl_crop, gt_unseen_compl_crop, 1)]

            pred_df = pd.concat([pred_df, compl_df], axis=1)

            # Nocs
            noc_df = pd.DataFrame()

            gt_scan_noc_crop = torch.unsqueeze(vg_crop(rpn_gt['bscan_nocs_mask'][B], pred_bbox), dim=0)
            gt_scan_inst_mask_crop = vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox)
            gt_scan_noc_inst_crop = ((gt_scan_inst_mask_crop == int(obj_idx)) & torch.all(gt_scan_noc_crop >= 0, 1))[0]#[0]

            gt_noc_on_gt_inst = gt_scan_noc_crop[0,:, gt_scan_noc_inst_crop].T

            if type(outputs['noc']) == dict:
                pred_noc_crop = outputs['noc']['noc_values'][B][j]
            else:
                pred_noc_crop = outputs['noc'][B][j]
            pred_noc_on_gt_inst = pred_noc_crop[0,:, gt_scan_noc_inst_crop].T

            noc_df['acc5'] = [l1_acc(pred_noc_on_gt_inst, gt_noc_on_gt_inst, 0.05)]
            noc_df['acc10'] = [l1_acc(pred_noc_on_gt_inst, gt_noc_on_gt_inst, 0.10)]
            noc_df['acc15'] = [l1_acc(pred_noc_on_gt_inst, gt_noc_on_gt_inst, 0.15)]

            if analyses is not None and 'noc' in analyses:
                for analysis_key, analysis_vals in analyses['noc'].items():
                    if 'rot_angle_diffs' == analysis_key:
                        for k,d in enumerate(['x','y','z']):
                            noc_df[analysis_key+'_%s'%d] = [analysis_vals[B][j][k].item()]
                    if 'transl_diffs' == analysis_key:
                        for k,d in enumerate(['x','y','z']):
                            noc_df[analysis_key + '_%s' % d] = [analysis_vals[B][j][k].item()]

                    if 'transl_diffs_center' == analysis_key:
                        for k,d in enumerate(['x','y','z']):
                            noc_df[analysis_key + '_%s' % d] = [analysis_vals[B][j][k].item()]
                    if 'scale_diffs' == analysis_key:
                        noc_df[analysis_key] = [analysis_vals[B][j].item()]

            pred_df = pd.concat([pred_df, noc_df], axis=1)

            dscan_df = pd.concat([dscan_df, pred_df], axis=0, ignore_index=True)

        bdscan_df = pd.concat([bdscan_df, dscan_df], axis=0, ignore_index=True)

    return bdscan_df, bdscan_gt_df