from __future__ import absolute_import, division, print_function

import sys, os
import math
import time, datetime
import numpy as np
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

class Trainer:

    def __init__(self, options):
        self.opt = options
        self.log_path = CONF.PATH.OUTPUT

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model setup --------------------------------------------------------------------------------------------------
        '''
        self.voxel_out_dim = 12
        self.edge_out_dim = 8
        self.models["voxel_encoder"] = networks.VoxelEncoder(input_channel=1, output_channel=self.voxel_out_dim)
        self.models["voxel_encoder"].to(self.device)

        if not self.opt.no_pose:
            classifier_in_dim = 2 * self.voxel_out_dim + self.edge_out_dim
        else:
            classifier_in_dim = 2 * self.voxel_out_dim

        self.models["edge_encoder"] = networks.MLP(7, [8, self.edge_out_dim], dropout_p=None, use_batchnorm=False)
        self.models["edge_encoder"].to(self.device)

        self.models["edge_classifier"] = networks.EdgeClassifier(input_dim=classifier_in_dim)
        self.models["edge_classifier"].to(self.device)

        init_weights(self.models["edge_classifier"], init_type='kaiming', init_gain=0.02)
        init_weights(self.models["voxel_encoder"], init_type='kaiming', init_gain=0.02)
        init_weights(self.models["edge_encoder"], init_type='kaiming', init_gain=0.02)

        self.parameters_to_train += list(self.models["voxel_encoder"].parameters())
        self.parameters_to_train += list(self.models["edge_classifier"].parameters())
        self.parameters_to_train += list(self.models["edge_encoder"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, weight_decay=self.opt.weight_decay)

        #self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 15, 0.5)
        
        self.classifier_dataset = {}

        # Loss Function ---------------------------------------------------------------------------------------------
        if self.opt.use_triplet:
            print('Using Triplet Loss for gradients ...')
            self.criterion = nn.TripletMarginLoss(margin=1.0, p=2.0, reduction='mean') # p=2 is euclidian dist, m=1 margin between anchor and negative sample
            self.loss_key = 'Triplet_loss'
        elif self.opt.use_l1:
            print('Using L1 Loss for gradients ...')
            self.criterion = nn.L1Loss(reduction='mean')
            self.loss_key = 'L1_loss'
        else:
            print('Using BCE Loss for gradients ...')
            self.loss_key = 'BCE_loss'
        '''
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

    def inference(self, vis_pose=False):
        """
        Run the entire inference pipeline
        """
        print("Starting inference and loading models ...")
        self.start_time = time.time()
        self.load_model()
        self.set_eval()

        overall_targets = []
        overall_predictions = []
        overall_gt_objects = 0
        overall_misses = 0
        overall_fps = 0

        for batch_idx, inputs in enumerate(self.test_loader):
            with torch.no_grad():
                outputs, _ = self.process_batch(inputs, mode='test')

            # Eval Metrics
            for idx, output in enumerate(outputs):
                overall_predictions.append(output['prediction'])
                overall_targets.append(output['target'])
                overall_gt_objects += output['total_gt_objs']
                overall_misses += output['misses']
                overall_fps += output['false_positives']

                if vis_pose:
                    visualise_tracking(inputs[idx], output)

            if int(batch_idx + 1) % 10 == 0:
                predictions = np.concatenate(overall_predictions)
                targets = np.concatenate(overall_targets)

                Prec = get_precision(predictions, targets)
                Rec = get_recall(predictions, targets)
                F1 = get_f1(predictions, targets)
                MOTA, _ = get_MOTA(predictions, targets, overall_gt_objects, overall_misses, overall_fps)

                print("[Batch Idx]: ", batch_idx + 1, "[MOTA]: ", MOTA, "[F1 score]: ", F1, "[Precision]: ", Prec,
                      "[Recall]: ", Rec)

        predictions = np.concatenate(overall_predictions)
        targets = np.concatenate(overall_targets)

        Prec = get_precision(predictions, targets)
        Rec = get_recall(predictions, targets)
        F1 = get_f1(predictions, targets)
        MOTA, id_switches = get_MOTA(predictions, targets, overall_gt_objects, overall_misses, overall_fps)

        print("Final Evaluation Scores :", "[MOTA]: ", MOTA, "[F1 score]: ", F1, "[Precision]: ", Prec, "[Recall]: ", Rec)
        print("Numbers :", "[Misses]: ", overall_misses, "[False Positives]: ", overall_fps, "[ID Switches]: ", id_switches)

    def run_epoch(self):
        """
        Run a single epoch of training and validation
        """
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            _, losses = self.process_batch(inputs, mode='train')

            loss = losses[self.loss_key]

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            self.opt.log_frequency = 1
            if int(batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, loss.cpu().data)
                self.log("train", losses)
            '''
            if int(batch_idx + 1) % int(round(len(self.train_loader)/1)) == 0: # validation n time per epoch
                self.val()
            '''
            self.step += 1
        #self.model_lr_scheduler.step()
        self.val()

    def process_batch(self, inputs, mode='train'):
        '''
        Process batch:
        1. Siamese Network encodes voxel grids in feature space and concat with Pose for object feature
        2. Get active/non-active edges (TARGETS) by computing IoU pred and GT and compare GT object ids
        -> obj1 img1 with all obj img2, obj2 img1 with all obj img2
        3. Concat object features between consecutive frames -> feature vector of 2 x 16 object features
        5. Edge classification into active non-active
        '''

        batch_loss = 0
        batch_size = len(inputs)
        batch_output = []

        for batch_idx, input in enumerate(inputs):

            graph_in_features = []
            num_imgs = len(input)
            total_gt_objs = 0
            total_pred_objs = 0
            misses = 0

            for i in range(num_imgs):

                # One voxel batch consists of all instances in one image
                voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                num_instances = int(voxels.shape[0])
                voxel_feature = torch.unsqueeze(self.models["voxel_encoder"](voxels.to(self.device)), dim=0) # 1 x num_instances x feature_dim

                if not self.opt.no_pose: # with pose
                    rot = torch.unsqueeze(input[i]['rotations'], dim=0)  # 1 x num_instances x 3
                    trans = torch.unsqueeze(input[i]['translations'], dim=0) # 1 x num_instances x 3
                    scale = torch.unsqueeze(torch.unsqueeze(input[i]['scales'], -1), dim=0) # 1 x num_instances x 1
                    pose = torch.cat((rot, trans, scale), dim=-1) # 1 x num_instance x 7
                    #print('Pose feature', pose.shape)

                    img_feat = torch.cat((voxel_feature, pose.to(self.device)), dim=-1) # 1 x num instances x 16
                else:
                    img_feat = voxel_feature

                graph_in_features.append(img_feat)

                per_img_gt_objs = int(input[i]['gt_object_id'].shape[-1])
                total_gt_objs += per_img_gt_objs # Number of ground truth objects in one frame
                total_pred_objs += num_instances # Number of predicted objects in one frame

                if num_instances < per_img_gt_objs: # Missing detections/ False Negatives per image or for a GT box no matching pred box found
                    misses += per_img_gt_objs - num_instances

            # Object Association
            scene_id = input[0]['scene'] + '_' + mode

            if scene_id not in self.classifier_dataset:
                classifier_data = construct_siamese_dataset(input, graph_in_features, thres=self.box_iou_thres)
                self.classifier_dataset[scene_id] = classifier_data
                edge_features = self.classifier_dataset[scene_id]['edge_features']
            else:
                try:
                    edge_features = recompute_edge_features(graph_in_features, self.classifier_dataset[scene_id]['obj_ids'])
                except:
                    print('ID issue :', scene_id)
                    traceback.print_exc()

            targets = self.classifier_dataset[scene_id]['targets']
            false_positives = self.classifier_dataset[scene_id]['false_positives']
            vis_idxs = self.classifier_dataset[scene_id]['vis_idxs']
            non_empty = False

            if self.opt.use_triplet:
                # Only used for triplet loss
                anchors = self.classifier_dataset[scene_id]['anchors']
                positive_samples = self.classifier_dataset[scene_id]['positive_samples']
                negative_samples = self.classifier_dataset[scene_id]['negative_samples']

                if anchors:
                    non_empty = True
                    anchors = torch.cat(anchors, dim=0)
                    positive_samples = torch.cat(positive_samples, dim=0)
                    negative_samples = torch.cat(negative_samples, dim=0)

            if edge_features:
                edge_feature = torch.cat(edge_features, dim=0)# num instance combinations in one sequence x 32
                edge_feature = compute_edge_emb(edge_feature, self.models['edge_encoder'], voxel_dim=self.voxel_out_dim)
            else:
                print('Empty tensor', ', Bad scene:', input[0]['scene'])
                batch_loss = 1
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence
            similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1) # n*m
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

            if self.opt.use_triplet and non_empty:
                losses = self.compute_triplet_loss(anchors, positive_samples, negative_samples) # shape n samples x 16
            elif self.opt.use_triplet:
                print('No triplet pairs found for sequence {}'.format(input[0]['scene']))
                losses = {}
                losses[self.loss_key] = 1
            else: # for BCE and L1 loss compute distance in edge prediction space
                losses = self.compute_losses(similarity_pred, targets)

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred)
            batch_loss += losses[self.loss_key] / batch_size
            outputs = {'total_gt_objs': total_gt_objs, 'false_positives': false_positives, 'misses': misses, 'vis_idxs': vis_idxs,
                       'prediction': similarity_pred.cpu().detach().numpy(), 'target': targets.cpu().detach().numpy()} # output per scene

            batch_output.append(outputs)

        losses[self.loss_key] = batch_loss
        return batch_output, losses

    def val(self):
        """
        Validate the model on the validation set
        Batch size 1
        """
        self.set_eval()

        print("Starting evaluation ...")
        val_loss = []

        overall_targets = []
        overall_predictions = []
        overall_gt_objects = 0
        overall_misses = 0
        overall_fps = 0


        for batch_idx, inputs in enumerate(self.val_loader):

            with torch.no_grad():
                outputs, losses = self.process_batch(inputs, mode='val')

            if isinstance(losses[self.loss_key], float):
                val_loss.append(losses[self.loss_key])
            else:
                val_loss.append(losses[self.loss_key].detach().cpu().item())

            # Eval Metrics
            for output in outputs:

                overall_predictions.append(output['prediction'])
                overall_targets.append(output['target'])
                overall_gt_objects += output['total_gt_objs']
                overall_misses += output['misses']
                overall_fps += output['false_positives']

            if int(batch_idx + 1) % 10 == 0:
                print("[Validation] Batch Idx: ", batch_idx + 1)

        predictions = np.concatenate(overall_predictions)
        targets = np.concatenate(overall_targets)

        Prec = get_precision(predictions, targets)
        Rec = get_recall(predictions, targets)
        F1 = get_f1(predictions, targets)
        MOTA, _ = get_MOTA(predictions, targets, overall_gt_objects, overall_misses, overall_fps)

        print('[Validation Loss]: ', np.array(val_loss).mean(), "[Precision]: ", Prec,
              "[Recall]: ", Rec, "[F1_score]: ", F1, "[MOTA]: ", MOTA)

        val_loss_mean = {self.loss_key: np.array(val_loss).mean(), 'Precision': Prec,
                         'Recall': Rec, 'F1_score': F1, 'MOTA': MOTA}

        self.log("val", val_loss_mean)

        self._save_valmodel(val_loss_mean)
        del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, targets):
        '''
        Balanced loss giving active and non-active edges same magnitude
        inputs: predictions
        '''

        losses = {}

        if self.opt.use_l1:
            l1_loss = self.criterion(torch.sigmoid(inputs), targets)
            losses[self.loss_key] = l1_loss
        else:
            num_active = torch.count_nonzero(targets)
            num_all = torch.numel(targets)
            pos_weight = (num_all - num_active) / num_active
            balanced_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean", pos_weight=pos_weight)

            losses[self.loss_key] = balanced_loss

        return losses

    def compute_triplet_loss(self, anchor, positive, negative):

        losses = {}

        triplet = self.criterion(anchor, positive, negative)
        losses[self.loss_key] = triplet

        return losses


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
