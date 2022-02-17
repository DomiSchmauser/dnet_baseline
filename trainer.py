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

from dvis import dvis

import networks
import datasets

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from model_cfg import init_cfg
from utils.train_utils import sec_to_hm_str, _sparsify, loss_to_logging
from utils.net_utils import vg_crop
from datasets.sequence_chunking import chunk_sequence, batch_collate

# Model import
from models.BCompletionDec2 import BCompletionDec2
from models.BDenseBackboneGeo import BDenseBackboneGeo
from models.BPureSparseBackbone import BPureSparseBackboneCol
from models.BSparseRPN_pure import BSparseRPN_pure
from models.BNocDec2 import BNocDec2
from models.BNocDec2_ume import BNocDec2_ume

class Trainer:

    def __init__(self, options):
        self.opt = options
        self.log_path = CONF.PATH.OUTPUT

        self.models = {}
        self.parameters_rpn = []
        self.parameters_general = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model setup --------------------------------------------------------------------------------------------------
        cfg = init_cfg()
        self.sparse_pretrain_ep = cfg['general']['sparse_pretrain_epochs']

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
        # Dataset ----------------------------------------------------------------------------------------------------
        DATA_DIR = CONF.PATH.FRONTDATA
        self.dataset = datasets.Front_dataset

        train_dataset = self.dataset(base_dir=DATA_DIR, split='train')

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
            num_workers=self.opt.num_workers,
            collate_fn=batch_collate,
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
            print('Start training ...')
            if self.epoch < self.sparse_pretrain_ep:
                sparse_pretrain = True
                print('Sparse network pretraining ...')
            else:
                sparse_pretrain = False
                print('Full pipeline training ...')

            self.run_epoch(sparse_pretrain=sparse_pretrain)
            if (self.epoch+1) % self.opt.save_frequency == 0 \
                    and self.opt.save_model and (self.epoch+1) >= self.opt.start_saving and not sparse_pretrain:
                self.save_model(is_val=False)

    def val(self, sparse_pretrain=False):
        self.set_eval()

        print("Starting evaluation ...")
        overall_losses = []
        for batch_idx, inputs in enumerate(self.val_loader):

            with torch.no_grad():
                outputs, losses, analyses, _ = self.validation_step(inputs, sparse_pretrain=sparse_pretrain)
                overall_losses.append(losses)

                if sparse_pretrain:
                    rpn_loss = losses['rpn']['total_loss']
                    losses['total_loss'] = rpn_loss.item()
                else:
                    rpn_loss = losses['rpn']['total_loss']
                    loss = losses['total_loss']
                    losses['total_loss'] = loss.item() + rpn_loss.item()
            '''
                eval_df, gt_eval_df = evaluate(outputs, inputs, losses, analyses)
                collection_eval_df: pd.DataFrame = pd.concat([collection_eval_df, eval_df], axis=0, ignore_index=True)
                collection_gt_eval_df: pd.DataFrame = pd.concat([collection_gt_eval_df, gt_eval_df], axis=0,
                                                                ignore_index=True)
                '''

        log_losses = loss_to_logging(overall_losses)
        self.log("val", log_losses)

        self._save_valmodel(log_losses['total_loss'])
        del inputs, outputs, losses, overall_losses

        self.set_train()

    def inference(self):
        """
        Run the entire inference pipeline
        """
        print("Starting inference and loading models ...")
        self.start_time = time.time()
        self.load_model()
        self.set_eval()

        for batch_idx, inputs in enumerate(self.test_loader):
            with torch.no_grad():
                outputs = self.infer_step(inputs)

    def run_epoch(self, sparse_pretrain=False):
        self.set_train()

        rotation_diff = []
        location_diff = []
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            _, losses, analyses, _ = self.training_step(inputs, sparse_pretrain=sparse_pretrain)

            if sparse_pretrain:
                rpn_loss = losses['rpn']['total_loss']
                self.rpn_optimizer.zero_grad()
                rpn_loss.backward()
                losses['total_loss'] = rpn_loss.item()  # release graph after backprop
                self.rpn_optimizer.step()
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

            self.opt.log_frequency = 1
            if int(batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses['total_loss'])

                log_losses = loss_to_logging([losses])
                self.log("train", log_losses)

            self.step += 1
        #self.model_lr_scheduler.step()
        if not sparse_pretrain:
            print('Mean Rotation Error: ', torch.mean(torch.cat(rotation_diff, dim=0), dim=0), 'Mean Translation Error :', torch.mean(torch.cat(location_diff, dim=0), dim=0)*0.03)
        self.val(sparse_pretrain=sparse_pretrain)

    def training_step(self, inputs, sparse_pretrain=False):
        '''
        One general training step of the whole network pipeline
        Inputs: Batch of num sequences
        '''

        total_loss = torch.cuda.FloatTensor([0])
        losses = dict()
        analyses = dict()

        # Unpack data
        dense_features, sparse_features, rpn_features = inputs

        rpn_gt = {}
        rpn_gt['breg_sparse'] = rpn_features[0]
        rpn_gt['scan_shape'] = dense_features.shape[2:]
        rpn_gt['bboxes'] = rpn_features[1]  # list(N boxes x 6)
        rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
        rpn_gt['bscan_inst_mask'] = rpn_features[3]  # list( 1 x X x Y x Z)
        rpn_gt['bscan_nocs_mask'] = rpn_features[4] # list( 3 x X x Y x Z)
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

    def validation_step(self, inputs, sparse_pretrain=False):
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

        rpn_gt = {}
        rpn_gt['breg_sparse'] = rpn_features[0]
        rpn_gt['scan_shape'] = dense_features.shape[2:]
        rpn_gt['bboxes'] = rpn_features[1]  # list(N boxes x 6)
        rpn_gt['bobj_idxs'] = rpn_features[2]  # list(list ids)
        rpn_gt['bscan_inst_mask'] = rpn_features[3]  # list( 1 x X x Y x Z)
        rpn_gt['bscan_nocs_mask'] = rpn_features[4]  # list( 3 x X x Y x Z)

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.infer_step(sparse_features)
        rpn_output = self.rpn.infer_step(s_e2, rpn_gt)
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output
        outputs["rpn"] = {"bbbox_lvl0": bbbox_lvl0, "bgt_target": bgt_target, "brpn_conf": brpn_conf}

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.infer_step(dense_features)

        completion_output = self.completion.infer_step(x_d2, x_e2, x_e1, bbbox_lvl0)
        outputs["completion"] = completion_output
        binst_occ = [[x.squeeze() for x in compls] for compls in outputs["completion"]]

        noc_output = self.noc_infer.infer_step(x_d2, x_e2, x_e1, rpn_gt, bbbox_lvl0, binst_occ)
        outputs["noc"] = noc_output

        return outputs

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
            if n == 'graph_net':
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
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
