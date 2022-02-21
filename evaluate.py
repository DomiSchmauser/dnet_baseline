import numpy as np
import torch


import logging
import re

import json
from datetime import datetime
#from utils.timer import Timer, AverageMeter
#from utils.model_storage import snapshot, store_history

from utils.net_utils import iou3d
from pathlib import Path

import pdb

from utils.net_utils import merge_dict, vg_crop, rotation_matrix_to_angle_axis
from typing import Optional, List


from dvis import dvis
import pandas as pd

from utils.eval_utils import eval_det_prematched
from utils.eval_utils import mask_iou, l1_acc, mpl_plot

log = logging.getLogger(__name__)


def evaluate(outputs, inputs, losses=None, analyses=None):

    return evaluate_bdscan(outputs, inputs, losses, analyses)

def evaluate_bdscan(outputs, inputs, losses=None, analyses=None):

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

    bbbox_lvl0, bgt_target, brpn_conf = outputs['rpn']['bbbox_lvl0'], outputs['rpn']['bgt_target'], outputs['rpn']['brpn_conf']
    bdscan_df = pd.DataFrame() 
    
    bdscan_gt_df = pd.DataFrame()
    for B in range(len(rpn_gt['bboxes'])):

        # Tracking infos
        seq_pattern = "val/(.*?)/coco_data"
        scan_pattern = "rgb_(.*?).png"
        seq_name = re.search(seq_pattern, bscan_info[B]).group(1)
        scan_idx = int(re.search(scan_pattern, bscan_info[B]).group(1))

        for obj_idx in rpn_gt['bobj_idxs'][B]:

            dscan_gt_df = pd.DataFrame()
            dscan_gt_df['seq_name'] = [seq_name]
            dscan_gt_df['scan_idx'] = [scan_idx]
            dscan_gt_df['obj_idx'] = [obj_idx]
            #dscan_gt_df['model_id'] = [dobj.model_id]
            dscan_gt_df['class_id'] = [bscan_obj[B][str(obj_idx)]['category_id']]
            dscan_gt_df['class_name'] = [bscan_obj[B][str(obj_idx)]['class_name']]
            dscan_gt_df['rot_sym'] = [bscan_obj[B][str(obj_idx)]['rot_sym']]
            dscan_gt_df['scan_coverage'] = [bscan_obj[B][str(obj_idx)]['num_occ']]

            bdscan_gt_df = pd.concat([bdscan_gt_df, dscan_gt_df], axis=0)
        


    for B in range(len(bbbox_lvl0)):
        dscan_df = pd.DataFrame() 
        scan_objs = bscan_obj[B]
        bbox_lvl0, gt_target, rpn_conf = bbbox_lvl0[B], bgt_target[B], brpn_conf[B]

        # Tracking infos
        seq_pattern =  "val/(.*?)/coco_data"
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

            
            #info_df['model_id'] = [dobject.model_id]
            info_df['class_name'] = [dobject['class_name']]
            #info_df['scan_coverage_amodal'] = [dobject.scan_coverage_amodal]

            pred_df = pd.concat([pred_df, info_df], axis=1, keys=['','info'])

            if 'rpn' in outputs:
                rpn_df = pd.DataFrame()
                # TODO MORE ADVANCED EVALUATION
                gt_bbox = torch.from_numpy(dobject['box_3d']).cuda().int()

                rpn_df['iou'] = [iou3d(pred_bbox.unsqueeze(0).float(), gt_bbox.unsqueeze(0).float()).item()]
                rpn_df['conf'] = [rpn_conf[j].item()]

                '''
                if 'rpn' in losses:
                    for loss_key, loss_vals in losses['rpn'].items():
                        if loss_key != 'total_loss':
                            rpn_df[loss_key] = [loss_vals[B].item()]
                '''
                
                rpn_df.columns = pd.MultiIndex.from_product([['rpn'], rpn_df.columns])
                pred_df = pd.concat([pred_df, rpn_df], axis=1)
            
            # Completion
            compl_df = pd.DataFrame()

            gt_scan_inst_mask_crop = vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox, crop_box=True )
            gt_scan_compl_crop = (gt_scan_inst_mask_crop == int(obj_idx))
            pred_compl_crop = outputs['completion'][B][j]

            gt_unseen_compl_crop = (torch.abs(vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox, crop_box=True )) > 0) & gt_scan_compl_crop
            pred_unseen_compl_crop = (torch.abs(vg_crop(rpn_gt['bscan_inst_mask'][B], pred_bbox, crop_box=True )) > 0) & pred_compl_crop

            #gt_seen_compl_crop = (torch.abs(vg_crop(dscan.tsdf_geo, pred_bbox, crop_box=True )) <= 0.4) & gt_scan_compl_crop
            #pred_seen_compl_crop = (torch.abs(vg_crop(dscan.tsdf_geo, pred_bbox, crop_box=True )) <= 0.4) & pred_compl_crop

            compl_df['iou'] = [mask_iou(pred_compl_crop, gt_scan_compl_crop, 1)]
            compl_df['unseen_iou'] = [mask_iou(pred_unseen_compl_crop, gt_unseen_compl_crop, 1)]
            #compl_df['seen_iou'] = [mask_iou(pred_seen_compl_crop, gt_seen_compl_crop, 1)]

            #for loss_key, loss_vals in losses['completion'].items():
            #    compl_df[loss_key] = [loss_vals[B][j % len(loss_vals[B])].item()]

            compl_df.columns = pd.MultiIndex.from_product([['completion'], compl_df.columns])
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

            #for loss_key, loss_vals in losses['noc'].items():
            #    noc_df[loss_key] = [loss_vals[B][j].item()]


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

            noc_df.columns = pd.MultiIndex.from_product([['noc'], noc_df.columns])
            pred_df = pd.concat([pred_df, noc_df], axis=1)

            dscan_df = pd.concat([dscan_df, pred_df], axis=0, ignore_index=True)


        bdscan_df = pd.concat([bdscan_df, dscan_df], axis=0, ignore_index=True)       
    # pair-wise results
    if 'fusion' in outputs:
        fusion_df = pd.DataFrame()

        # per matching sequence
        for seq_name, ppairs in outputs['fusion'].items():
            
            for ppair in ppairs:
                seq_fusion_df = pd.DataFrame()
                seq_fusion_df['seq_name'] = [seq_name]
                # just use A->B 
                seq_fusion_df['scan_idx'] = [ppair.A.scan_idx]
                seq_fusion_df['prop_idx'] = [ppair.A.prop_idx]

                # TODO SEE IF USEFUL
                """
                seq_fusion_df['match_scan_idx'] = [ppair.B.scan_idx]
                seq_fusion_df['match_prop_idx'] = [ppair.B.prop_idx]
                """
                #seq_fusion_df['gt_worldA2worldB'] = [ppair.gt_worldA2worldB]
                #seq_fusion_df['pred_worldA2worldB'] = [ppair.pred_worldA2worldB]

                gt_rel_world_rot = rotation_matrix_to_angle_axis(ppair.gt_worldA2worldB[:3,:3].unsqueeze(0))[0].cpu().numpy()
                pred_rel_world_rot = rotation_matrix_to_angle_axis(ppair.pred_worldA2worldB[:3,:3].unsqueeze(0))[0].cpu().numpy()

                del_rel_world_rot = rotation_matrix_to_angle_axis((ppair.gt_worldA2worldB[:3,:3] @ ppair.pred_worldA2worldB[:3,:3].T).unsqueeze(0))[0].cpu().numpy()

                ppair.gt_worldA2worldB = ppair.gt_worldA2worldB.cpu().numpy()
                ppair.pred_worldA2worldB = ppair.pred_worldA2worldB.cpu().numpy()

                ppair.gt_worldA2worldB_t = ppair.gt_worldA2worldB_t.cpu().numpy()
                ppair.pred_worldA2worldB_t = ppair.pred_worldA2worldB_t.cpu().numpy()
                
                """
                seq_fusion_df['gt_rel_world_trans_x'] = [ppair.gt_worldA2worldB_t[0]]
                seq_fusion_df['gt_rel_world_trans_y'] = [ppair.gt_worldA2worldB_t[1]]
                seq_fusion_df['gt_rel_world_trans_z'] = [ppair.gt_worldA2worldB_t[2]]

                seq_fusion_df['pred_rel_world_trans_x'] = [ppair.pred_worldA2worldB_t[0]]
                seq_fusion_df['pred_rel_world_trans_y'] = [ppair.pred_worldA2worldB_t[1]]
                seq_fusion_df['pred_rel_world_trans_z'] = [ppair.pred_worldA2worldB_t[2]]

                seq_fusion_df['gt_rel_world_rot_x'] = [np.abs(gt_rel_world_rot[0]) * 180 / np.pi]
                seq_fusion_df['gt_rel_world_rot_y'] = [np.abs(gt_rel_world_rot[1]) * 180 / np.pi]
                seq_fusion_df['gt_rel_world_rot_z'] = [np.abs(gt_rel_world_rot[2]) * 180 / np.pi]

                seq_fusion_df['pred_rel_world_rot_x'] = [np.abs(pred_rel_world_rot[0]) * 180 / np.pi]
                seq_fusion_df['pred_rel_world_rot_y'] = [np.abs(pred_rel_world_rot[1]) * 180 / np.pi]
                seq_fusion_df['pred_rel_world_rot_z'] = [np.abs(pred_rel_world_rot[2]) * 180 / np.pi]
                """

                seq_fusion_df['del_abs_rel_world_trans_x'] = [np.abs(ppair.gt_worldA2worldB_t[0] - ppair.pred_worldA2worldB_t[0])]
                seq_fusion_df['del_abs_rel_world_trans_y'] = [np.abs(ppair.gt_worldA2worldB_t[1] - ppair.pred_worldA2worldB_t[1])]
                seq_fusion_df['del_abs_rel_world_trans_z'] = [np.abs(ppair.gt_worldA2worldB_t[2] - ppair.pred_worldA2worldB_t[2])]

                seq_fusion_df['del_rel_world_rot_x'] = [np.abs(del_rel_world_rot) * 180 / np.pi]
                seq_fusion_df['del_rel_world_rot_y'] = [np.abs(del_rel_world_rot) * 180 / np.pi]
                seq_fusion_df['del_rel_world_rot_z'] = [np.abs(del_rel_world_rot) * 180 / np.pi]
                """
                seq_fusion_df['del_rel_world_rot_x'] = [np.abs((gt_rel_world_rot[0] - pred_rel_world_rot[0]) % (2*np.pi)) * 180 / np.pi]
                seq_fusion_df['del_rel_world_rot_y'] = [np.abs((gt_rel_world_rot[1] - pred_rel_world_rot[1])  % (2*np.pi)) * 180 / np.pi]
                seq_fusion_df['del_rel_world_rot_z'] = [np.abs((gt_rel_world_rot[2] - pred_rel_world_rot[2])  % (2*np.pi)) * 180 / np.pi]
                """



                seq_fusion_df.columns = pd.MultiIndex.from_product([['fusion'], seq_fusion_df.columns])
                fusion_df = pd.concat([fusion_df, seq_fusion_df], axis=0)
        # join with bdscan_df
        if len(fusion_df)> 0:
            bdscan_df = pd.merge(bdscan_df, fusion_df, how='left', left_on=[('info','seq_name'), ('info', 'scan_idx'), ('info', 'prop_idx')], right_on=[('fusion', 'seq_name'), ('fusion', 'scan_idx'), ('fusion', 'prop_idx')])
            bdscan_df.drop([('fusion', 'seq_name'), ('fusion', 'scan_idx'), ('fusion', 'prop_idx')], axis=1, inplace=True)

    return bdscan_df, bdscan_gt_df

def evaluate_dscan(outputs, dscan, losses=None, analyses=None):
    dscan_df = pd.DataFrame() # columns=['seq_name', 'scan_idx', 'obj_idx', 'model_id', 'class_name']
    # dscan_df.append({'seq_name': 2, 'scan_idx': 30, "model_id": 'ff', 'momo': 30}, ignore_index=True)
    bbox_lvl0, gt_target = outputs['rpn']['bbox_lvl0'], outputs['rpn']['gt_target']

    # object level evaluation
    for j, (pred_bbox, obj_idx) in enumerate(zip(bbox_lvl0, gt_target)):
        dobject = dscan.objects[obj_idx]
        pred_df = pd.DataFrame()
        info_df = pd.DataFrame()

        info_df['seq_name'] = [dscan.seq_name]
        info_df['scan_idx'] = [dscan.scan_idx]
        info_df['model_id'] = [dobject.model_id]
        info_df['class_name'] = [dobject.class_name]
        info_df['scan_coverage_amodal'] = [dobject.scan_coverage_amodal]

        pred_df = pd.concat([pred_df, info_df], axis=1, keys=['','info'])

        if 'rpn' in outputs:
            rpn_df = pd.DataFrame()
            # TODO
        
        if 'completion' in outputs:
            compl_df = pd.DataFrame()

            gt_scan_inst_mask_crop = vg_crop(dscan.scan_inst_mask, pred_bbox)
            gt_scan_compl_crop = (gt_scan_inst_mask_crop == int(obj_idx))
            pred_compl_crop = outputs['completion'][j]

            gt_unseen_compl_crop = (torch.abs(vg_crop(dscan.tsdf_geo, pred_bbox)) > 0.4) & gt_scan_compl_crop
            pred_unseen_compl_crop = (torch.abs(vg_crop(dscan.tsdf_geo, pred_bbox)) > 0.4) & pred_compl_crop

            gt_seen_compl_crop = (torch.abs(vg_crop(dscan.tsdf_geo, pred_bbox)) <= 0.4) & gt_scan_compl_crop
            pred_seen_compl_crop = (torch.abs(vg_crop(dscan.tsdf_geo, pred_bbox)) <= 0.4) & pred_compl_crop

            compl_df['iou'] = [mask_iou(pred_compl_crop, gt_scan_compl_crop, 1)]
            compl_df['unseen_iou'] = [mask_iou(pred_unseen_compl_crop, gt_unseen_compl_crop, 1)]
            compl_df['seen_iou'] = [mask_iou(pred_seen_compl_crop, gt_seen_compl_crop, 1)]

            for loss_key, loss_vals in losses['completion'].items():
                compl_df[loss_key] = [loss_vals[j].item()]

            compl_df.columns = pd.MultiIndex.from_product([['completion'], compl_df.columns])
            pred_df = pd.concat([pred_df, compl_df], axis=1)


        if 'noc' in outputs:
            noc_df = pd.DataFrame()

            gt_scan_noc_crop = vg_crop(dscan.scan_noc, pred_bbox)
            gt_scan_inst_mask_crop = vg_crop(dscan.scan_inst_mask, pred_bbox)
            gt_scan_noc_inst_crop = ((gt_scan_inst_mask_crop == int(obj_idx)) & torch.all(gt_scan_noc_crop >= 0, 1))[0][0]

            gt_noc_on_gt_inst = gt_scan_noc_crop[0,:, gt_scan_noc_inst_crop].T

            if 'noc_values' in outputs['noc']:
                pred_noc_crop = outputs['noc']['noc_values'][j]
            else:
                pred_noc_crop = outputs['noc'][j]
            pred_noc_on_gt_inst = pred_noc_crop[0,:, gt_scan_noc_inst_crop].T

            noc_df['acc5'] = [l1_acc(pred_noc_on_gt_inst, gt_noc_on_gt_inst, 0.05)]
            noc_df['acc10'] = [l1_acc(pred_noc_on_gt_inst, gt_noc_on_gt_inst, 0.10)]
            noc_df['acc15'] = [l1_acc(pred_noc_on_gt_inst, gt_noc_on_gt_inst, 0.15)]

            for loss_key, loss_vals in losses['noc'].items():
                noc_df[loss_key] = [loss_vals[j].item()]

            if 'noc' in analyses:
                for analysis_key, analysis_vals in analyses['noc'].items():
                    if analysis_key[0] == '$':
                        noc_df[analysis_key[1:]] = [analysis_vals[j].item()]
                    if analysis_key[0] == '&':
                        for i in range(len(analysis_vals[j])):
                            noc_df[analysis_key[1:] + "_%s"%(['x','y','z'][i])] = [analysis_vals[j][i].item()]

            noc_df.columns = pd.MultiIndex.from_product([['noc'], noc_df.columns])
            pred_df = pd.concat([pred_df, noc_df], axis=1)
        
        dscan_df = pd.concat([dscan_df, pred_df], axis=0, ignore_index=True)
    
    # scan level evaluation
    num_predictions = len(bbox_lvl0)
    if 'scan_completion' in outputs:
        scan_completion_df = pd.DataFrame()
        gt_scan_compl = torch.abs(dscan.scan_sdf) < 2
        pred_scan_compl = outputs['scan_completion']

        gt_unseen_scan_compl = (torch.abs(dscan.tsdf_geo)> 0.4) & gt_scan_compl
        pred_unseen_scan_compl = (torch.abs(dscan.tsdf_geo)> 0.4) & pred_scan_compl

        gt_seen_scan_compl = (torch.abs(dscan.tsdf_geo)<= 0.4) & gt_scan_compl
        pred_seen_scan_compl = (torch.abs(dscan.tsdf_geo)<= 0.4) & pred_scan_compl
        
        #repeat for each prediction
        scan_completion_df['iou'] = num_predictions * [mask_iou(pred_scan_compl, gt_scan_compl, 1)]
        scan_completion_df['unseen_iou'] = num_predictions * [mask_iou(pred_unseen_scan_compl, gt_unseen_scan_compl, 1)]
        scan_completion_df['seen_iou'] = num_predictions * [mask_iou(pred_seen_scan_compl, gt_seen_scan_compl, 1)]
        
        for loss_key, loss_val in losses['scan_completion'].items():
                scan_completion_df[loss_key] = num_predictions * [loss_val.item()]

        scan_completion_df.columns = pd.MultiIndex.from_product([['scan_completion'], scan_completion_df.columns])
        dscan_df = pd.concat([dscan_df, scan_completion_df], axis=1)
                    

    return dscan_df
        



def df_to_ap(df, gt_df, writer_step=None):
    
    pred_df = df.loc[:,[('info', 'seq_name'), ('info', 'scan_idx'), ('info', 'prop_idx'), ('info', 'gt_target'), ('info', 'model_id'), ('info', 'class_name'), ('rpn', 'iou'), ('rpn', 'conf')]]
    # remove duplicates in case of overfitting
    pred_df.drop_duplicates(subset=[('info', 'seq_name'), ('info', 'scan_idx'), ('info', 'prop_idx')], keep='first', inplace=True)
    gt_df.drop_duplicates(subset=['seq_name', 'scan_idx', 'obj_idx'], keep='first', inplace=True)
    # get prec, recall, AP for detection
    
    det_pred_dict = dict()
    for name, group in pred_df[[('info', 'seq_name'), ('info', 'scan_idx'), ('info', 'gt_target'),  ('rpn', 'conf'), ('rpn', 'iou')]].groupby([('info', 'seq_name'), ('info', 'scan_idx')]):
        det_pred_dict[name] = list(zip(group[('info', 'gt_target')], group[('rpn', 'conf')], group[('rpn', 'iou')]))
    
    det_gt_dict = dict()
    for name, group in gt_df.groupby(['seq_name', 'scan_idx']):
        det_gt_dict[name] = list(group['obj_idx'])
    


    det_rec25, det_pre25, det_ap25, conf25 = eval_det_prematched(det_pred_dict, det_gt_dict, ovthresh=0.25)
    det_rec50, det_pre50, det_ap50, conf50 = eval_det_prematched(det_pred_dict, det_gt_dict, ovthresh=0.50)

    # log directly additional plots

    writer, global_step = writer_step

    writer.add_figure('Precision_25', mpl_plot(conf25, det_pre25, 'confidence', 'precision_25', col='g'), global_step)
    writer.add_figure('Precision_50', mpl_plot(conf50, det_pre50, 'confidence', 'precision_50', col='g'), global_step)
    writer.add_figure('Recall_25', mpl_plot(conf25, det_rec25, 'confidence', 'recall_25', col='b'), global_step)
    writer.add_figure('Recall_50', mpl_plot(conf50, det_rec50, 'confidence', 'recall_50', col='b'), global_step)

    writer.add_figure('PR_25', mpl_plot(det_rec25, det_pre25, 'recall', 'precision', col='h'), global_step)
    writer.add_figure('PR_50', mpl_plot(det_rec50, det_pre50, 'recall', 'precision', col='h'), global_step)

    return pd.DataFrame({'det_ap25': [det_ap25], 'det_ap50': [det_ap50]})
    

def df_to_ap_compl(df, gt_df, writer_step=None):
    
    pred_df = df.loc[:,[('info', 'seq_name'), ('info', 'scan_idx'), ('info', 'prop_idx'), ('info', 'gt_target'), ('info', 'model_id'), ('info', 'class_name'), ('completion', 'iou'), ('completion', 'conf')]]
    # remove duplicates in case of overfitting
    pred_df.drop_duplicates(subset=[('info', 'seq_name'), ('info', 'scan_idx'), ('info', 'prop_idx')], keep='first', inplace=True)
    gt_df.drop_duplicates(subset=['seq_name', 'scan_idx', 'obj_idx'], keep='first', inplace=True)
    # get prec, recall, AP for detection
    
    det_pred_dict = dict()
    for name, group in pred_df[[('info', 'seq_name'), ('info', 'scan_idx'), ('info', 'gt_target'),  ('completion', 'conf'), ('completion', 'iou')]].groupby([('info', 'seq_name'), ('info', 'scan_idx')]):
        det_pred_dict[name] = list(zip(group[('info', 'gt_target')], group[('completion', 'conf')], group[('completion', 'iou')]))
    
    det_gt_dict = dict()
    for name, group in gt_df.groupby(['seq_name', 'scan_idx']):
        det_gt_dict[name] = list(group['obj_idx'])
    


    det_rec25, det_pre25, det_ap25, conf25 = eval_det_prematched(det_pred_dict, det_gt_dict, ovthresh=0.25)
    det_rec50, det_pre50, det_ap50, conf50 = eval_det_prematched(det_pred_dict, det_gt_dict, ovthresh=0.50)



    return pd.DataFrame({'det_ap25': [det_ap25], 'det_ap50': [det_ap50]})
    

    

def kpi(df, gt_df, agg_lvl='collection', writer_step=None):
    kpi_df = pd.DataFrame()
    if len(df) == 0:
        return kpi_df
    if 'rpn' in df.columns.levels[0]:
        rpn_df = pd.DataFrame(df['rpn'].mean()).transpose()
        rpn_df.columns = pd.MultiIndex.from_product([['rpn'], rpn_df.columns])
        kpi_df = pd.concat([kpi_df, rpn_df], axis=1)
        
        # ap evaluation
        ap_df = df_to_ap(df, gt_df, writer_step)
        ap_df.columns = pd.MultiIndex.from_product([['rpn'], ap_df.columns])
        kpi_df = pd.concat([kpi_df, ap_df], axis=1)
        
    
    if 'completion' in df.columns.levels[0]:
        compl_df = pd.DataFrame(df['completion'].mean()).transpose()
        compl_df.columns = pd.MultiIndex.from_product([['completion'], compl_df.columns])
        kpi_df = pd.concat([kpi_df, compl_df], axis=1)
        
        # ap evaluation
        ap_df_compl = df_to_ap_compl(df, gt_df, writer_step)
        ap_df_compl.columns = pd.MultiIndex.from_product([['completion'], ap_df_compl.columns])
        kpi_df = pd.concat([kpi_df, ap_df_compl], axis=1)

    if 'noc' in df.columns.levels[0]:
        noc_df = pd.DataFrame(df['noc'].median()).transpose()
        noc_df.columns = pd.MultiIndex.from_product([['noc'], noc_df.columns])
        kpi_df = pd.concat([kpi_df, noc_df], axis=1)
    
    # scene level aggregation (take unique value per seq_name/scan_idx pair)
    if 'scan_completion' in df.columns.levels[0]:
        scan_completion_df = pd.DataFrame(df[['info', 'scan_completion']].groupby([('info', 'seq_name'), ('info', 'scan_idx')]).max()['scan_completion'].mean()).transpose()
        scan_completion_df.columns = pd.MultiIndex.from_product([['scan_completion'], scan_completion_df.columns])
        kpi_df = pd.concat([kpi_df, scan_completion_df], axis=1)

    if 'fusion' in df.columns.levels[0]:
        fusion_df = pd.DataFrame(df['fusion'].median(skipna=True)).transpose()

        fusion_df.columns = pd.MultiIndex.from_product([['fusion'], fusion_df.columns])
        kpi_df = pd.concat([kpi_df, fusion_df], axis=1)

    return kpi_df
    

    
    


            