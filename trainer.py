from __future__ import absolute_import, division, print_function

import sys, os
import math
import time, datetime
import numpy as np
import pandas as pd
import json
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn import functional as F
import MinkowskiEngine as ME
import open3d as o3d
from pathlib import Path
import re

from dvis import dvis

import networks
import datasets

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from model_cfg import init_cfg
from utils.train_utils import sec_to_hm_str, loss_to_logging
from utils.net_utils import vg_crop
from utils.scan_merge import merge2seq
from datasets.sequence_chunking import batch_collate, batch_collate_infer, batch_collate_cpu
from evaluate import evaluate, get_mota

# Model import
from models.BCompletionDec2 import BCompletionDec2
from models.BDenseBackboneGeo import BDenseBackboneGeo
from models.BPureSparseBackbone import BPureSparseBackboneCol
from models.BSparseRPN_pure import BSparseRPN_pure
from models.BNocDec2 import BNocDec2
from models.BNocDec2_ume import BNocDec2_ume

# Tracking import
from tracking.tracking_front import Tracker
from utils.visualise import visualise_pred_sequence

class Trainer:

    def __init__(self, options):
        self.opt = options
        self.log_path = CONF.PATH.OUTPUT
        self.experiment_dir = CONF.PATH.STORAGE

        self.models = {}
        self.parameters_rpn = []
        self.parameters_general = []
        self.no_crash_mode = self.opt.no_crash_mode
        #self.no_crash_mode = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model setup --------------------------------------------------------------------------------------------------
        cfg = init_cfg()
        self.sparse_pretrain_ep = cfg['general']['sparse_pretrain_epochs']
        self.dense_pretrain_ep = cfg['general']['dense_pretrain_epochs']
        self.overfit = cfg['general']['overfit']
        self.quantization_size = cfg['general']['quantization_size']

        # Sparse Backbone & RPN
        self.models["sparse_backbone"] = networks.PureSparseBackboneCol_Res1(conf=cfg['sparse_backbone'])
        self.models["sparse_backbone"].to(self.device)
        self.sparse_backbone = BPureSparseBackboneCol(cfg['sparse_backbone'], self.models['sparse_backbone'])

        self.models["rpn"] = networks.SparseRPNNet4_Res1(conf=cfg['rpn'])
        self.models["rpn"].to(self.device)
        self.rpn = BSparseRPN_pure(cfg['rpn'], self.models['rpn'])

        self.parameters_rpn += list(self.models["sparse_backbone"].parameters())
        self.parameters_rpn += list(self.models["rpn"].parameters())

        # Dense Backbone, Completion, Nocs
        self.models["dense_backbone"] = networks.DenseBackboneEPND2(conf=cfg['dense_backbone'])
        self.models["dense_backbone"].to(self.device)
        self.backbone = BDenseBackboneGeo(cfg['dense_backbone'], self.models['dense_backbone'])

        self.models["completion"] = networks.DenseCompletionDec2Bigger(conf=cfg['completion'])
        self.models["completion"].to(self.device)
        self.completion = BCompletionDec2(cfg['completion'], self.models['completion'])

        self.models["nocs"] = networks.DenseNocDec2(conf=cfg['nocs'])
        self.models["nocs"].to(self.device)
        self.noc = BNocDec2(cfg['nocs'], self.models['nocs'])

        self.noc_infer = BNocDec2_ume(cfg['nocs'], self.models['nocs'])


        self.parameters_general += list(self.models["dense_backbone"].parameters())
        self.parameters_general += list(self.models["completion"].parameters())
        self.parameters_general += list(self.models["nocs"].parameters())

        #init_weights(self.models["edge_classifier"], init_type='kaiming', init_gain=0.02)
        #self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 15, 0.5)

        # Optimizer --------------------------------------------------------------------------------------------------
        self.rpn_optimizer = optim.Adam(self.parameters_rpn, self.opt.learning_rate_rpn,
                                          weight_decay=self.opt.weight_decay)

        self.general_optimizer = optim.Adam(self.parameters_general, self.opt.learning_rate_general,
                                        weight_decay=self.opt.weight_decay)
        # Tracking ---------------------------------------------------------------------------------------------------
        self.Tracker = Tracker()
        # Dataset ----------------------------------------------------------------------------------------------------
        DATA_DIR = CONF.PATH.FRONTDATA
        self.dataset = datasets.Front_dataset

        train_dataset = self.dataset(base_dir=DATA_DIR, split='train', overfit=self.overfit)

        if self.opt.num_workers > 0:
            self.train_loader = DataLoader(
                train_dataset,
                self.opt.batch_size,
                shuffle=True,
                num_workers=self.opt.num_workers,
                collate_fn=batch_collate_cpu,
                pin_memory=True,
                drop_last=True)
        else:
            self.train_loader = DataLoader(
                train_dataset,
                self.opt.batch_size,
                shuffle=True,
                num_workers=self.opt.num_workers,
                collate_fn=batch_collate,
                pin_memory=False,
                drop_last=True)


        val_dataset = self.dataset(
            base_dir=DATA_DIR,
            split='val')

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=batch_collate_infer,
            pin_memory=False,
            drop_last=False)

        test_dataset = self.dataset(
            base_dir=DATA_DIR,
            split='test')

        self.infer_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=batch_collate_infer,
            pin_memory=False,
            drop_last=False)

        if not os.path.exists(self.opt.log_dir):
            os.makedirs(self.opt.log_dir)

        self.writers = {}
        for mode in ["train", "val"]:
            logging_path = os.path.join(self.opt.log_dir, mode)
            self.writers[mode] = SummaryWriter(logging_path)

        num_train_samples = len(train_dataset)
        num_eval_samples = len(val_dataset)
        print("There are {} training images and {} validation images in total...".format(num_train_samples,
                                                                                       num_eval_samples))
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
        Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        if self.opt.resume:
            print('Loading pretrained models and resume training...')
            self.load_model()

        for self.epoch in range(self.opt.num_epochs):

            if self.epoch < self.sparse_pretrain_ep:
                sparse_pretrain = True
                dense_pretrain = False
                print('Sparse network pretraining ...')
            elif self.epoch < self.sparse_pretrain_ep + self.dense_pretrain_ep:
                sparse_pretrain = False
                dense_pretrain = True
                print('Dense network pretraining ...')
            else:
                sparse_pretrain = False
                dense_pretrain = False
                print('Full pipeline training ...')

            self.run_epoch(sparse_pretrain=sparse_pretrain, dense_pretrain=dense_pretrain)
            if (self.epoch+1) % self.opt.save_frequency == 0:
                self.save_model(is_val=False)

    def val(self, sparse_pretrain=False, dense_pretrain=False):
        self.set_eval()

        print("Starting evaluation ...")
        overall_losses = []
        rotation_diff = []
        location_diff = []
        for batch_idx, inputs in enumerate(self.val_loader):

            if int(batch_idx + 1) % 50 == 0 and not sparse_pretrain and not dense_pretrain:
                print('Batch {} of {} Batches'.format(int((batch_idx+1)), int(len(self.val_loader))))
                print('Mean Rotation Error: ', torch.median(torch.cat(rotation_diff, dim=0), dim=0).values,
                      'Mean Translation Error :',
                      torch.median(torch.cat(location_diff, dim=0), dim=0).values * self.quantization_size)


            with torch.no_grad():
                if self.no_crash_mode:
                    try:
                        outputs, losses, analyses, _ = self.validation_step(inputs, sparse_pretrain=sparse_pretrain, dense_pretrain=dense_pretrain)
                    except:
                        traceback.print_exc()
                        continue
                else:
                    outputs, losses, analyses, _ = self.validation_step(inputs, sparse_pretrain=sparse_pretrain, dense_pretrain=dense_pretrain)

                overall_losses.append(losses)

                if sparse_pretrain and not dense_pretrain:
                    rpn_loss = losses['rpn']['total_loss']
                    losses['total_loss'] = rpn_loss.item()
                elif not sparse_pretrain and dense_pretrain:
                    loss = losses['total_loss']
                    losses['total_loss'] = loss.item()
                else:
                    rpn_loss = losses['rpn']['total_loss']
                    loss = losses['total_loss']
                    losses['total_loss'] = loss.item() + rpn_loss.item()

                    for b_idx, scan_rot_diff in enumerate(analyses['noc']['rot_angle_diffs']):
                        for inst_idx in range(len(scan_rot_diff)):
                            rotation_diff.append(torch.unsqueeze(scan_rot_diff[inst_idx].detach().cpu(), dim=0))
                            location_diff.append(
                                torch.unsqueeze(analyses['noc']['transl_diffs'][b_idx][inst_idx].detach().cpu(), dim=0))


        log_losses = loss_to_logging(overall_losses)
        self.log("val", log_losses)

        if not sparse_pretrain and not dense_pretrain:
            print('Mean Rotation Error: ', torch.median(torch.cat(rotation_diff, dim=0), dim=0).values, 'Mean Translation Error :', torch.median(torch.cat(location_diff, dim=0), dim=0).values*self.quantization_size)

        if not sparse_pretrain and not dense_pretrain:
            self._save_valmodel(log_losses['total_loss'])
        del inputs, outputs, losses, overall_losses

        self.set_train()

    def inference(self, store_results=False, vis=False, mota_log_freq=250, get_pose_error=False, pose_only=False, resume_chkpt=False, vis_pose=True, seq_len=25, classwise=True):
        """
        Run the entire inference pipeline and perform tracking afterwards
        mota_log_freq/ 25 = num_sequences per logging
        """
        print("Starting inference and loading models ...")
        self.start_time = time.time()
        self.load_model()
        self.set_eval()
        self.no_crash_mode = False

        if get_pose_error:
            self.val(sparse_pretrain=False, dense_pretrain=False)

        collection_eval_df = pd.DataFrame()
        collection_gt_eval_df = pd.DataFrame()
        mota_df = pd.DataFrame()
        classes_df = {
            1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame(),
            4: pd.DataFrame(), 5: pd.DataFrame(),
            6: pd.DataFrame(), 7: pd.DataFrame()
        }
        classes_iou = {
            1: [], 2: [], 3: [],
            4: [], 5: [],
            6: [], 7: []
        }

        overall_iou = []

        occ_grids = dict()

        rotation_errors = []
        location_errors = []

        if resume_chkpt:
            mota_df = pd.read_excel('mota.xlsx')
            start_idx = mota_df.last_valid_index()
            mota_df = mota_df.iloc[: , 1:]


        for batch_idx, inputs in enumerate(self.infer_loader):


            if int(batch_idx + 1) % 100 == 0:
                print('Sequence {} of {} Sequences'.format(int((batch_idx+1)/25), int(len(self.infer_loader)/25)))

            if vis_pose and batch_idx == 0:
                # Only single scene
                vis_pc = inputs[2][-1][0]

            with torch.no_grad():
                outputs, bscan_info = self.infer_step(inputs)

                # Skipping invalid images
                if outputs is None:
                    continue

                if vis and int(batch_idx + 1) % 10 == 0:
                    dvis(torch.squeeze(inputs[2][3][0] > 0), fmt='voxels')
                    boxes = outputs['rpn']['bbbox_lvl0'][0]
                    for box in boxes:
                        dvis(torch.unsqueeze(box, dim=0), fmt='box', c=3)

                if pose_only:
                    # Rotation Location error logging
                    rot_err, loc_err = torch.cat(outputs['errors'][0][0], dim=0), torch.cat(outputs['errors'][1][0], dim=0)
                    rotation_errors.append(rot_err)
                    location_errors.append(loc_err)

                else:
                    eval_df, gt_eval_df = evaluate(outputs, inputs, None, None)
                    collection_eval_df: pd.DataFrame = pd.concat([collection_eval_df, eval_df], axis=0, ignore_index=True)
                    collection_gt_eval_df: pd.DataFrame = pd.concat([collection_gt_eval_df, gt_eval_df], axis=0,
                                                                    ignore_index=True)

                    # Scan level GT occupancy grids
                    for B, grid in enumerate(inputs[0]):
                        seq_pattern = "test/(.*?)/coco_data"
                        scan_pattern = "rgb_(.*?).png"
                        seq_name = re.search(seq_pattern, bscan_info[B]).group(1)
                        scan_idx = int(re.search(scan_pattern, bscan_info[B]).group(1))

                        if seq_name not in occ_grids:
                            occ_grids[seq_name] = dict()
                            grid = torch.squeeze(grid).detach().cpu().numpy()
                            occ_grids[seq_name][scan_idx] = grid
                        else:
                            grid = torch.squeeze(grid).detach().cpu().numpy()
                            occ_grids[seq_name][scan_idx] = grid

                # Evaluate Tracking per Sequence for fixed sequence lenght == 25
                if int(batch_idx + 1) % seq_len == 0:
                    if pose_only:
                        print('Rotation error :', torch.median(torch.cat(rotation_errors, dim=0), dim=0).values)
                        print('Location error :', torch.median(torch.cat(location_errors, dim=0), dim=0).values * self.quantization_size)
                        continue

                    if len(occ_grids[seq_name]) != seq_len:
                        print('skipping sequence !!!')
                        continue
                    # Sort and rearrange df
                    collection_gt_eval_df.sort_values(by='seq_name')
                    collection_gt_eval_df.sort_values(by='scan_idx')
                    collection_eval_df.sort_values(by='seq_name')
                    collection_eval_df.sort_values(by='scan_idx')
                    sequences = collection_gt_eval_df['seq_name'].unique().tolist()
                    #assert len(sequences) == 1

                    if int(batch_idx + 1) % 2500 == 0:
                        # Create a Pandas Excel writer using XlsxWriter as the engine.
                        writer = pd.ExcelWriter('mota_100.xlsx', engine='xlsxwriter')
                        mota_df.to_excel(writer, sheet_name='mota')
                        writer.save()

                        for cls, cls_df in classes_df.items():
                            writer = pd.ExcelWriter('mota_'+str(cls)+'.xlsx', engine='xlsxwriter')
                            cls_df.to_excel(writer, sheet_name='mota')
                            writer.save()

                    if len(sequences) != 1:
                        # Create a Pandas Excel writer using XlsxWriter as the engine.
                        writer = pd.ExcelWriter('mota.xlsx', engine='xlsxwriter')
                        mota_df.to_excel(writer, sheet_name='mota')
                        writer.save()
                        continue

                    for seq_idx, seq in enumerate(sequences):
                        gt_seq_df = collection_gt_eval_df.loc[collection_gt_eval_df['seq_name'] == seq]
                        pred_seq_df = collection_eval_df.loc[collection_eval_df['seq_name'] == seq]

                        #todo Get 3D IoU Classwise and overall
                        iou_all = pred_seq_df['compl_iou'].to_numpy()
                        iou_all[iou_all != np.array(None)]
                        nan_array = np.isnan(iou_all)
                        not_nan_array = ~ nan_array
                        iou_all = iou_all[not_nan_array]
                        iou_all_mean = iou_all.mean()
                        overall_iou.append(iou_all_mean)
                        for iou_cls, cls_list in classes_iou.items():
                            iou_cls_tables = pred_seq_df[pred_seq_df['class_id'] == iou_cls]
                            if iou_cls_tables.empty:
                                continue
                            iou_cls_val = iou_cls_tables['compl_iou'].to_numpy()
                            iou_cls_val[iou_cls_val != np.array(None)]
                            nan_array = np.isnan(iou_cls_val)
                            not_nan_array = ~ nan_array
                            iou_cls_val = iou_cls_val[not_nan_array]
                            iou_cls_val = iou_cls_val.mean()
                            cls_list.append(iou_cls_val)

                        if vis_pose:
                            pred_trajectories, gt_trajectories, seq_data = self.Tracker.analyse_trajectories_vis(gt_seq_df, pred_seq_df, occ_grids[seq])
                            #visualise_gt_sequence(gt_trajectories, seq_name='Vis_Seq', pc=vis_pc, grid=occ_grids[seq_name])
                            visualise_pred_sequence(pred_trajectories, seq_name='Vis_Seq', pc=vis_pc, seq_len=seq_len)
                            continue


                        pred_trajectories, gt_trajectories, seq_data = self.Tracker.analyse_trajectories(gt_seq_df, pred_seq_df, occ_grids[seq])
                        gt_traj_tables = self.Tracker.get_traj_tables(gt_trajectories, seq_data, 'gt')
                        pred_traj_tables = self.Tracker.get_traj_tables(pred_trajectories, seq_data, 'pred')
                        seq_mota_summary, mot_events = self.Tracker.eval_mota(pred_traj_tables, gt_traj_tables)
                        mota_df = pd.concat([mota_df, seq_mota_summary], axis=0, ignore_index=True)

                        if classwise:
                            # Just run eval MOTA on all classes dfs and aggregate seperately
                            for cls, cls_df in classes_df.items():
                                gt_cls_traj_tables = gt_traj_tables[gt_traj_tables['obj_cls'] == cls]
                                if gt_cls_traj_tables.empty:
                                    continue

                                # Get assignments
                                matches = mot_events[mot_events['Type'] == 'MATCH']
                                class_ids = gt_cls_traj_tables['obj_idx'].unique()
                                filtered_matched = matches[matches['HId'].isin(class_ids)]  # all mate
                                frame_idxs = filtered_matched.index.droplevel(1)
                                obj_idxs = filtered_matched['HId']
                                fp_cls_traj_tables = pred_traj_tables.loc[pred_traj_tables['scan_idx'].isin(frame_idxs) & pred_traj_tables['obj_idx'].isin(obj_idxs)]
                                pred_cls_traj_tables = pred_traj_tables[pred_traj_tables['obj_cls'] == cls]
                                pred_merge_table = pd.concat([fp_cls_traj_tables, pred_cls_traj_tables]).drop_duplicates()
                                pred_merge_table['gt_obj_idx'] = pred_merge_table['gt_obj_idx'].apply(np.array)
                                class_mota_summary, _ = self.Tracker.eval_mota(pred_merge_table, gt_cls_traj_tables)
                                classes_df[cls] = pd.concat([cls_df, class_mota_summary], axis=0, ignore_index=True)

                    # Cleanup space
                    collection_eval_df = pd.DataFrame()
                    collection_gt_eval_df = pd.DataFrame()
                    occ_grids = dict()

                # Logging
                if int(batch_idx + 1) % mota_log_freq == 0 and not pose_only and not vis_pose:
                    mota_score = mota_df.loc[:, 'mota'].mean(axis=0)
                    Prec = mota_df.loc[:, 'precision'].mean(axis=0) #How many of found are correct
                    Rec = mota_df.loc[:, 'recall'].mean(axis=0) #How many predictions found
                    num_misses = mota_df.loc[:, 'num_misses'].sum(axis=0)
                    num_false_positives = mota_df.loc[:, 'num_false_positives'].sum(axis=0)
                    id_switches = mota_df.loc[:, 'num_switches'].sum(axis=0)
                    num_objects_gt = mota_df.loc[:, 'num_objects'].sum(axis=0)
                    mota_accumulated = get_mota(num_objects_gt, num_misses, num_false_positives, id_switches)
                    print('Accumulated MOTA:', mota_accumulated, ' Averaged MOTA:', mota_score,
                          ' Precision:', Prec,
                          ' Recall:', Rec,
                          'ID switches:', id_switches,
                          ' Current sum Misses:', num_misses,
                          ' Current sum False Positives:', num_false_positives)

                    # Logging
                    if int(batch_idx + 1) % mota_log_freq*2 == 0 and not pose_only and not vis_pose and classwise:
                        cls_mapping = {
                            1: 'chair', 2: 'table', 3: 'sofa',
                            4: 'bed', 5: 'tv_stand',
                            6: 'cooler', 7: 'night_stand'
                        }
                        if classwise:
                            for cls, cls_df in classes_df.items():
                                if cls_df.empty:
                                    continue
                                cls_mota_accumulated = get_mota(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                                   cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                                   cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                                   cls_df.loc[:, 'num_switches'].sum(axis=0))
                                print('Class MOTA :', cls_mapping[cls], 'score:', cls_mota_accumulated)

                            for cls, cls_list in classes_iou.items():
                                if not cls_list:
                                    continue
                                cls_iou_score = np.array(cls_list).mean()
                                print('Class IoU :', cls_mapping[cls], 'score:', cls_iou_score)

                            print('Overall IoU', np.array(overall_iou).mean())
        # Final Logging
        print('Final tracking scores :')
        mota_score = mota_df.loc[:, 'mota'].mean(axis=0)
        Prec = mota_df.loc[:, 'precision'].mean(axis=0)
        Rec = mota_df.loc[:, 'recall'].mean(axis=0)
        num_misses = mota_df.loc[:, 'num_misses'].sum(axis=0)
        num_false_positives = mota_df.loc[:, 'num_false_positives'].sum(axis=0)
        id_switches = mota_df.loc[:, 'num_switches'].sum(axis=0)
        num_objects_gt = mota_df.loc[:, 'num_objects'].sum(axis=0)
        mota_accumulated = get_mota(num_objects_gt, num_misses, num_false_positives, id_switches)
        print('Accumulated MOTA:', mota_accumulated, ' Averaged MOTA:', mota_score,
              ' Precision:', Prec,
              ' Recall:', Rec,
              'ID switches:', id_switches,
              ' Current sum Misses:', num_misses,
              ' Current sum False Positives:', num_false_positives)

        cls_mapping = {
            1: 'chair', 2: 'table', 3: 'sofa',
            4: 'bed', 5: 'tv_stand',
            6: 'cooler', 7: 'night_stand'
        }
        if classwise:
            for cls, cls_df in classes_df.items():
                if cls_df.empty:
                    continue
                cls_mota_accumulated = get_mota(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                cls_df.loc[:, 'num_switches'].sum(axis=0))
                print('Class MOTA :', cls_mapping[cls], 'score:', cls_mota_accumulated)

            for cls, cls_list in classes_iou.items():
                if not cls_list:
                    continue
                cls_iou_score = np.array(cls_list).mean()
                print('Class IoU :', cls_mapping[cls], 'score:', cls_iou_score)

            print('Overall IoU', np.array(overall_iou).mean())

        # Results to CSV
        if store_results:
            # store evaluations
            eval_dir = Path(self.experiment_dir, 'evaluations', 'inference')
            os.makedirs(eval_dir, exist_ok=True)

            collection_eval_df.to_csv(Path(eval_dir, 'collection.csv'), index=False)
            collection_gt_eval_df.to_csv(Path(eval_dir, 'collection_gt.csv'), index=False)

    def run_epoch(self, sparse_pretrain=False, dense_pretrain=False):
        self.set_train()

        rotation_diff = []
        location_diff = []
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # Training loop
            if self.no_crash_mode:
                try:
                    _, losses, analyses, _ = self.training_step(inputs, sparse_pretrain=sparse_pretrain,
                                                           dense_pretrain=dense_pretrain)
                except:
                    traceback.print_exc()
                    continue
            else:
                _, losses, analyses, _ = self.training_step(inputs, sparse_pretrain=sparse_pretrain, dense_pretrain=dense_pretrain)

            if sparse_pretrain and not dense_pretrain:
                rpn_loss = losses['rpn']['total_loss']
                self.rpn_optimizer.zero_grad()
                rpn_loss.backward()
                losses['total_loss'] = rpn_loss.item()  # release graph after backprop
                self.rpn_optimizer.step()

            elif not sparse_pretrain and dense_pretrain:
                loss = losses['total_loss']
                self.general_optimizer.zero_grad()
                loss.backward()
                losses['total_loss'] = loss.item()   # release graph after backprop
                self.general_optimizer.step()

            else:
                rpn_loss = losses['rpn']['total_loss']
                loss = losses['total_loss']

                self.rpn_optimizer.zero_grad()
                self.general_optimizer.zero_grad()

                rpn_loss.backward()
                loss.backward()

                losses['total_loss'] = loss.item() + rpn_loss.item()  # release graph after backprop

                for b_idx, scan_rot_diff in enumerate(analyses['noc']['rot_angle_diffs']):
                    for inst_idx in range(len(scan_rot_diff)):
                        rotation_diff.append(torch.unsqueeze(scan_rot_diff[inst_idx].detach().cpu(), dim=0))
                        location_diff.append(torch.unsqueeze(analyses['noc']['transl_diffs'][b_idx][inst_idx].detach().cpu(), dim=0))

                self.rpn_optimizer.step()
                self.general_optimizer.step()

            torch.cuda.empty_cache()

            # logging
            duration = time.time() - before_op_time

            self.opt.log_frequency = 5
            if int(batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses['total_loss'])

                log_losses = loss_to_logging([losses])
                self.log("train", log_losses)

            self.step += 1
        #self.model_lr_scheduler.step()

        if not sparse_pretrain and not dense_pretrain:
            print('Mean Rotation Error: ', torch.median(torch.cat(rotation_diff, dim=0), dim=0), 'Mean Translation Error :', torch.median(torch.cat(location_diff, dim=0), dim=0).values*self.quantization_size)

        self.val(sparse_pretrain=sparse_pretrain, dense_pretrain=dense_pretrain)

    def training_step(self, inputs, sparse_pretrain=False, dense_pretrain=False):
        '''
        One general training step of the whole network pipeline
        Inputs: Batch of num sequences
        '''

        total_loss = torch.cuda.FloatTensor([0])
        losses = dict()
        analyses = dict()

        # Unpack data everything
        dense_features, sparse_features, rpn_features = inputs

        if self.opt.num_workers > 0:

            # Move to GPU
            dense_features = dense_features.to(self.device)
            sparse_features = ME.SparseTensor(sparse_features[0].to(self.device), # 0 are features
                                              ME.utils.batched_coordinates(sparse_features[1]).to(self.device)) # 1 are coords
            #sparse_features = sparse_features.to(self.device)

            rpn_gt = {}
            rpn_gt['breg_sparse'] = ME.SparseTensor(rpn_features[0][0].to(self.device),
                                                    ME.utils.batched_coordinates(rpn_features[0][1]).to(self.device))
            rpn_gt['scan_shape'] = dense_features.shape[2:]
            boxes_gpu = []
            inst_mask_gpu = []
            nocs_mask_gpu = []
            for B in range(len(rpn_features[1])):
                boxes_gpu.append(rpn_features[1][B].to(self.device))
                inst_mask_gpu.append(rpn_features[3][B].to(self.device))
                nocs_mask_gpu.append(rpn_features[4][B].to(self.device))

            rpn_gt['bboxes'] = boxes_gpu  # list(N boxes x 6)
            rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
            rpn_gt['bscan_inst_mask'] = inst_mask_gpu  # list( 1 x X x Y x Z)
            rpn_gt['bscan_nocs_mask'] = nocs_mask_gpu # list( 3 x X x Y x Z)
            bscan_obj = rpn_features[5]

        else:
            rpn_gt = {}
            rpn_gt['breg_sparse'] = rpn_features[0]
            rpn_gt['scan_shape'] = dense_features.shape[2:]
            rpn_gt['bboxes'] = rpn_features[1]  # list(N boxes x 6)
            rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
            rpn_gt['bscan_inst_mask'] = rpn_features[3]  # list( 1 x X x Y x Z)
            rpn_gt['bscan_nocs_mask'] = rpn_features[4]  # list( 3 x X x Y x Z)
            bscan_obj = rpn_features[5]

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.training_step(sparse_features)
        rpn_output, rpn_losses, rpn_analyses, rpn_timings = self.rpn.training_step(s_e2, rpn_gt)
        losses["rpn"] = rpn_losses
        losses["rpn"]["total_loss"] = torch.mean(torch.cat(rpn_losses["bweighted_loss"], 0))
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output

        # Only train sparse pipeline
        if sparse_pretrain:
            return None, losses, rpn_analyses, {}

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.training_step(dense_features)  # enc_layer_1, enc_layer_2, dec_layer_2

        # Targets
        btarget_occ, bbbox_lvl0_compl, bgt_target_compl = [], [], []
        for B, (bbox_lvl0, gt_target) in enumerate(zip(bbbox_lvl0, bgt_target)):
            inst_crops = vg_crop(rpn_gt['bscan_inst_mask'][B], bbox_lvl0)

            target_num_occs = [bscan_obj[B][str(obj_idx)]['num_occ'] for obj_idx in gt_target] # number of occupancies per object
            inst_occ_targets = [(inst_crop == int(obj_idx)).unsqueeze(0) for obj_idx, inst_crop in
                                zip(gt_target, inst_crops)]
            valid_inst_occ_target = []
            valid_bbox_lvl0 = []

            valid_gt_target_compl = []

            for j, (inst_occ_target, target_num_occ, obj_idx) in enumerate(
                    zip(inst_occ_targets, target_num_occs, gt_target)):
                if inst_occ_target.sum() > target_num_occ * 0.05:
                    valid_inst_occ_target.append(inst_occ_target)
                    valid_bbox_lvl0.append(bbox_lvl0[j])
                    valid_gt_target_compl.append(obj_idx)
            btarget_occ.append(valid_inst_occ_target)
            bbbox_lvl0_compl.append(valid_bbox_lvl0)
            bgt_target_compl.append(valid_gt_target_compl)

        bbbox_lvl0_compl_s = [torch.stack(bboxes, 0) if len(bboxes) > 0 else [] for bboxes in bbbox_lvl0_compl]

        # Completion
        completion_output, completion_losses, completion_analyses, completion_timings = self.completion.training_step(
            x_d2, x_e2, x_e1, rpn_gt, bbbox_lvl0_compl_s, bgt_target_compl
        )
        losses["completion"] = completion_losses
        total_loss += torch.mean(torch.cat(completion_losses["bweighted_loss"], 0))

        # Nocs
        noc_output, noc_losses, noc_analyses, noc_timings = self.noc.training_step(
            x_d2, x_e2, x_e1, rpn_gt, bbbox_lvl0_compl_s, bgt_target_compl, bscan_obj
        )
        losses["noc"] = noc_losses
        analyses["noc"] = noc_analyses
        total_loss += torch.mean(torch.cat(noc_losses["bweighted_loss"], 0))

        losses["total_loss"] = total_loss

        return None, losses, analyses, {}

    def validation_step(self, inputs, sparse_pretrain=False, dense_pretrain=False):
        '''
        One general validation step of the whole network pipeline
        '''
        total_loss = 0
        losses = dict()
        analyses = dict()
        outputs = dict()

        # Unpack data
        dense_features, sparse_features, rpn_features = inputs

        rpn_gt = {}
        rpn_gt['breg_sparse'] = rpn_features[0]
        rpn_gt['scan_shape'] = dense_features.shape[2:]
        rpn_gt['bboxes'] = rpn_features[1]  # list(N boxes x 6)
        rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
        rpn_gt['bscan_inst_mask'] = rpn_features[3]  # list( 1 x X x Y x Z)
        rpn_gt['bscan_nocs_mask'] = rpn_features[4]  # list( 3 x X x Y x Z)
        bscan_obj = rpn_features[5]

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.validation_step(sparse_features)
        rpn_output, rpn_losses, rpn_analyses, rpn_timings = self.rpn.validation_step(s_e2, rpn_gt)
        losses["rpn"] = rpn_losses
        losses["rpn"]["total_loss"] = torch.mean(torch.cat(rpn_losses["bweighted_loss"], 0))
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output

        if sparse_pretrain:
            return None, losses, rpn_analyses, {}

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.validation_step(dense_features)

        # Targets
        btarget_occ, bgt_target_compl = [], []
        for B, (bbox_lvl0, gt_target) in enumerate(zip(bbbox_lvl0, bgt_target)):
            inst_crops = vg_crop(rpn_gt['bscan_inst_mask'][B], bbox_lvl0)
            inst_occ_targets = [(inst_crop == int(obj_idx)).unsqueeze(0) for obj_idx, inst_crop in
                                zip(gt_target, inst_crops)]
            btarget_occ.append(inst_occ_targets)
            bgt_target_compl.append(gt_target)

        # Completion
        completion_output, completion_losses, completion_analyses, completion_timings = self.completion.validation_step(
            x_d2, x_e2, x_e1, rpn_gt, bbbox_lvl0, bgt_target_compl
        )
        losses["completion"] = completion_losses
        total_loss += torch.mean(torch.cat(completion_losses["bweighted_loss"]))
        outputs["completion"] = completion_output

        # Nocs
        binst_occ = [[x.squeeze() for x in compls] for compls in outputs["completion"]]
        noc_output, noc_losses, noc_analyses, noc_timings = self.noc.validation_step(
            x_d2, x_e2, x_e1, rpn_gt, bbbox_lvl0, bgt_target_compl, bscan_obj, binst_occ=binst_occ
        )
        losses["noc"] = noc_losses
        analyses["noc"] = noc_analyses
        total_loss += torch.mean(torch.cat(noc_losses["bweighted_loss"]))

        outputs["noc"] = noc_output

        losses["total_loss"] = total_loss

        return outputs, losses, analyses, {}

    def infer_step(self, inputs):
        '''
        One general inference step of the whole network pipeline
        '''
        outputs = dict()

        # Unpack data
        dense_features, sparse_features, rpn_features = inputs
        if dense_features is None:
            return None, None

        rpn_gt = {}
        rpn_gt['breg_sparse'] = rpn_features[0]
        rpn_gt['scan_shape'] = dense_features.shape[2:]
        rpn_gt['bboxes'] = rpn_features[1]  # list(N boxes x 6)
        rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
        rpn_gt['bscan_inst_mask'] = rpn_features[3]  # list( 1 x X x Y x Z)
        rpn_gt['bscan_nocs_mask'] = rpn_features[4]  # list( 3 x X x Y x Z)
        bscan_obj = rpn_features[5]

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.infer_step(sparse_features)
        rpn_output = self.rpn.infer_step(s_e2, rpn_gt)
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output
        outputs["rpn"] = {"bbbox_lvl0": bbbox_lvl0, "bgt_target": bgt_target, "brpn_conf": brpn_conf}

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.infer_step(dense_features)

        completion_output = self.completion.infer_step(x_d2, x_e2, x_e1, bbbox_lvl0)
        outputs["completion"] = completion_output
        binst_occ = [[x.squeeze() for x in compls] for compls in outputs["completion"]] #dim WxHxL

        noc_output = self.noc_infer.infer_step(x_d2, x_e2, x_e1, rpn_gt, bgt_target, bscan_obj, bbbox_lvl0, binst_occ=binst_occ)
        outputs["noc"] = noc_output[0]
        outputs['errors'] = noc_output[1]

        return outputs, rpn_features[6]

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, is_val=True):
        """Save model weights """
        if is_val:
            best_folder = os.path.join(self.log_path, "best_model")
            if not os.path.exists(best_folder):
                os.makedirs(best_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(best_folder, "{}.pth".format(model_name + '_best'))
                to_save = model.state_dict()
                torch.save(to_save, save_path)

            if self.epoch >= self.opt.start_saving_optimizer:
                save_path_rpn = os.path.join(best_folder, "{}.pth".format("adam_best_rpn"))
                torch.save(self.rpn_optimizer.state_dict(), save_path_rpn)
                save_path_general = os.path.join(best_folder, "{}.pth".format("adam_best_general"))
                torch.save(self.general_optimizer.state_dict(), save_path_general)

        else:
            save_folder = os.path.join(self.log_path, "models", "epoch_{}".format(self.epoch+1))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.state_dict()
                torch.save(to_save, save_path)

            if self.epoch >= self.opt.start_saving_optimizer:
                save_path_rpn = os.path.join(save_folder, "{}.pth".format("adam_best_rpn"))
                torch.save(self.rpn_optimizer.state_dict(), save_path_rpn)
                save_path_general = os.path.join(save_folder, "{}.pth".format("adam_best_general"))
                torch.save(self.general_optimizer.state_dict(), save_path_general)

    def _save_valmodel(self, losses):

        mean_loss = losses

        json_path = os.path.join(self.opt.log_dir, 'val_metrics.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
                min_loss = min(data.values())
        else:
            data = {}
            min_loss = math.inf

        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        data.update({dt_string: mean_loss})
        now = None

        with open(json_path, 'w+') as f:
            json.dump(data, f)

        # save best model
        if mean_loss < min_loss and self.epoch >= self.opt.start_saving:
            print('Current Model Loss is lower than previous model, saving ...')
            self.save_model(is_val=True)


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("Loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading rpn adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam_rpn.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.rpn_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

        # loading general adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam_general.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.general_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx + 1, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
