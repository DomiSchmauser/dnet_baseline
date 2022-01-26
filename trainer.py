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

import networks
import datasets

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from model_cfg import init_cfg
from utils.train_utils import sec_to_hm_str
from utils.net_utils import vg_crop

# Model import
from models.BCompletionDec2 import BCompletionDec2
from models.BDenseBackboneGeo import BDenseBackboneGeo
from models.BPureSparseBackbone import BPureSparseBackboneCol
from models.BSparseRPN_pure import BSparseRPN_pure
from models.BNocDec2 import BNocDec2

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


        self.parameters_general += list(self.models["dense_backbone"].parameters())
        self.parameters_general += list(self.models["completion"].parameters())
        self.parameters_general += list(self.models["nocs"].parameters())




        #init_weights(self.models["edge_classifier"], init_type='kaiming', init_gain=0.02)
        #init_weights(self.models["voxel_encoder"], init_type='kaiming', init_gain=0.02)
        #init_weights(self.models["edge_encoder"], init_type='kaiming', init_gain=0.02)
       

        #self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 15, 0.5)

        # Loss Function ---------------------------------------------------------------------------------------------

        # Optimizer --------------------------------------------------------------------------------------------------
        self.rpn_optimizer = optim.Adam(self.parameters_rpn, self.opt.learning_rate,
                                          weight_decay=self.opt.weight_decay)

        self.general_optimizer = optim.Adam(self.parameters_general, self.opt.learning_rate,
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
            collate_fn=lambda x:x,
            pin_memory=True,
            drop_last=True)

        val_dataset = self.dataset(
            base_dir=DATA_DIR,
            split='val')

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            collate_fn=lambda x:x,
            pin_memory=True,
            drop_last=False)

        if not os.path.exists(self.opt.log_dir):
            os.makedirs(self.opt.log_dir)

        self.writers = {}
        for mode in ["train", "val"]:
            logging_path = os.path.join(self.opt.log_dir, mode)
            self.writers[mode] = SummaryWriter(logging_path)

        num_train_samples = len(train_dataset)
        num_eval_samples = len(val_dataset)
        print("There are {} training sequences and {} validation sequences in total...".format(num_train_samples,
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
            self.run_epoch()
            if (self.epoch+1) % self.opt.save_frequency == 0 \
                    and self.opt.save_model and (self.epoch+1) >= self.opt.start_saving:
                self.save_model(is_val=False)

    def val(self):
        self.set_eval()

        print("Starting evaluation ...")

        for batch_idx, inputs in enumerate(self.val_loader):

            with torch.no_grad():
                outputs, losses, analyses, _ = self.validation_step(inputs)

                '''
                eval_df, gt_eval_df = evaluate(outputs, inputs, losses, analyses)
                collection_eval_df: pd.DataFrame = pd.concat([collection_eval_df, eval_df], axis=0, ignore_index=True)
                collection_gt_eval_df: pd.DataFrame = pd.concat([collection_gt_eval_df, gt_eval_df], axis=0,
                                                                ignore_index=True)
                '''

        #self.log("val", val_loss_mean)

        #self._save_valmodel(val_loss_mean)
        del inputs, outputs, losses

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

    def run_epoch(self):
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            _, losses, analyses, _ = self.training_step(inputs)
            rpn_loss = losses['rpn']['total_loss']
            loss = losses['total_loss']

            self.rpn_optimizer.zero_grad()
            self.general_optimizer.zero_grad()

            rpn_loss.backward()
            loss.backward()

            losses['total_loss'] = loss.item() + rpn_loss.item()  # release graph after backprop

            self.rpn_optimizer.step()
            self.general_optimizer.step()

            torch.cuda.empty_cache()

            # logging
            duration = time.time() - before_op_time

            self.opt.log_frequency = 1
            if int(batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses['total_loss'])
                self.log("train", losses)

            self.step += 1
        #self.model_lr_scheduler.step()
        self.val()

    def training_step(self, inputs):
        '''
        One general training step of the whole network pipeline
        '''
        total_loss = torch.cuda.FloatTensor([0])
        losses = dict()
        analyses = dict()

        bdscan = inputs

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.training_step(bdscan)
        rpn_output, rpn_losses, rpn_analyses, rpn_timings = self.rpn.training_step(s_e2, bdscan)
        losses["rpn"] = rpn_losses
        losses["rpn"]["total_loss"] = torch.mean(torch.cat(rpn_losses["bweighted_loss"], 0))
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.training_step(bdscan)

        # Targets
        btarget_occ, bbbox_lvl0_compl, bgt_target_compl = [], [], []
        btarget_noc = []
        for B, (bbox_lvl0, gt_target) in enumerate(zip(bbbox_lvl0, bgt_target)):
            inst_crops = vg_crop(bdscan.bscan_inst_mask[B], bbox_lvl0)

            target_num_occs = [(torch.abs(bdscan.bobjects[B][obj_idx].sdf) < 2).sum() for obj_idx in gt_target]
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

            target_noc = []
            if len(valid_bbox_lvl0) > 0:
                for bbox in valid_bbox_lvl0:
                    target_noc.append(vg_crop(bdscan.bscan_noc[B], bbox))
            btarget_noc.append(target_noc)

        bbbox_lvl0_compl_s = [torch.stack(bboxes, 0) if len(bboxes) > 0 else [] for bboxes in bbbox_lvl0_compl]

        # Completion
        completion_output, completion_losses, completion_analyses, completion_timings = self.completion.training_step(
            x_d2, x_e2, x_e1, bdscan, bbbox_lvl0_compl_s, bgt_target_compl
        )
        losses["completion"] = completion_losses
        total_loss += torch.mean(torch.cat(completion_losses["bweighted_loss"], 0))

        # Nocs
        noc_output, noc_losses, noc_analyses, noc_timings = self.noc.training_step(
            x_d2, x_e2, x_e1, bdscan, bbbox_lvl0_compl_s, bgt_target_compl
        )
        losses["noc"] = noc_losses
        analyses["noc"] = noc_analyses
        total_loss += torch.mean(torch.cat(noc_losses["bweighted_loss"], 0))

        losses["total_loss"] = total_loss

        return None, losses, analyses, {}

    def validation_step(self, inputs):
        '''
        One general validation step of the whole network pipeline
        '''
        total_loss = 0
        losses = dict()
        analyses = dict()
        outputs = dict()

        bdscan = inputs

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.validation_step(bdscan)
        rpn_output, rpn_losses, rpn_analyses, rpn_timings = self.rpn.validation_step(s_e2, bdscan)
        losses["rpn"] = rpn_losses
        losses["rpn"]["total_loss"] = torch.mean(torch.cat(rpn_losses["bweighted_loss"], 0))
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.validation_step(bdscan)

        # Targets
        btarget_occ, bgt_target_compl = [], []
        btarget_noc = []
        for B, (bbox_lvl0, gt_target) in enumerate(zip(bbbox_lvl0, bgt_target)):
            inst_crops = vg_crop(bdscan.bscan_inst_mask[B], bbox_lvl0)
            target_num_occs = [(torch.abs(bdscan.bobjects[B][obj_idx].sdf) < 2).sum() for obj_idx in gt_target]
            inst_occ_targets = [(inst_crop == int(obj_idx)).unsqueeze(0) for obj_idx, inst_crop in
                                zip(gt_target, inst_crops)]
            btarget_occ.append(inst_occ_targets)
            bgt_target_compl.append(gt_target)


            target_noc = []
            for bbox in bbox_lvl0:
                target_noc.append(vg_crop(bdscan.bscan_noc[B], bbox))
            btarget_noc.append(target_noc)

        # Completion
        completion_output, completion_losses, completion_analyses, completion_timings = self.completion.validation_step(
            x_d2, x_e2, x_e1, bdscan, bbbox_lvl0, bgt_target_compl
        )
        losses["completion"] = completion_losses
        total_loss += torch.mean(torch.cat(completion_losses["bweighted_loss"]))
        outputs["completion"] = completion_output

        # Nocs
        binst_occ = [[x.squeeze() for x in compls] for compls in outputs["completion"]]
        noc_output, noc_losses, noc_analyses, noc_timings = self.noc.validation_step(
            x_d2, x_e2, x_e1, bdscan, bbbox_lvl0, bgt_target_compl, binst_occ
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

        bdscan = inputs

        # Sparse Pipeline ---------------------------------------------------------------------------------------------
        s_e2 = self.sparse_backbone.infer_step(bdscan)
        rpn_output = self.rpn.infer_step(s_e2, bdscan)
        bbbox_lvl0, bgt_target, brpn_conf = rpn_output
        outputs["rpn"] = {"bbbox_lvl0": bbbox_lvl0, "bgt_target": bgt_target, "brpn_conf": brpn_conf}

        # Dense Pipeline ----------------------------------------------------------------------------------------------
        x_e1, x_e2, x_d2 = self.backbone.infer_step(bdscan)

        completion_output = self.completion.infer_step(x_d2, x_e2, x_e1, bbbox_lvl0)
        outputs["completion"] = completion_output
        binst_occ = [[x.squeeze() for x in compls] for compls in outputs["completion"]]

        noc_output = self.noc.infer_step(x_d2, x_e2, x_e1, bdscan, bbbox_lvl0, binst_occ)
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
                save_path = os.path.join(best_folder, "{}.pth".format("adam_best"))
                torch.save(self.model_optimizer.state_dict(), save_path)

        else:
            save_folder = os.path.join(self.log_path, "models", "epoch_{}".format(self.epoch+1))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.state_dict()
                torch.save(to_save, save_path)

            if self.epoch >= self.opt.start_saving_optimizer:
                save_path = os.path.join(save_folder, "{}.pth".format("adam"))
                torch.save(self.model_optimizer.state_dict(), save_path)

    def _save_valmodel(self, losses):

        mean_loss = losses[self.loss_key]

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
