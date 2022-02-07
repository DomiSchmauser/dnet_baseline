import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.net_utils import vg_crop, hmgt, get_scale, kabsch_trans_rot, rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix, get_aligned2noc
from utils.net_utils import cats
from typing import Optional, List

from trimesh.transformations import rotation_matrix, translation_matrix, scale_matrix
from dvis import dvis

class BNocDec2(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net
        self.bbox_shape = np.array(conf['bbox_shape'])
        self.gt_augm = conf['gt_augm']
        self.crit = nn.L1Loss().cuda()
        self.noc_weight = conf['weights']['noc']
        self.rot_weight = conf['weights']['rot']
        self.transl_weight = conf['weights']['transl']
        self.noc_samples = conf['noc_samples']

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
             
        

    def validation_step(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bdscan, bbbox_lvl0: List, bgt_target: List, binst_occ=None):
        if not all([len(bbox_lvl0)>0 for bbox_lvl0 in bbbox_lvl0]):
            return [[]], {'bweighted_loss': [torch.Tensor([0.0]).cuda()], 'bnoc_gt_inst_loss': [torch.Tensor([0.0]).cuda()], 'brot_gt_inst_loss': [torch.Tensor([0.0]).cuda()]}, {}, {}
        # rotate predicted NOCS based on gt object rotation
        # x_d0_crops optimize to fit gt nocs rotated by best_rot_angle_y
        # Kabsch( R @ NOC, ref) = R.T @ Kabsch(NOC, ref)over
        # rotate pred instead of gt: 
        # pred_NOC = (Rbest) @ gt_NOCS
        '''
        bbest_rots = [[angle_axis_to_rotation_matrix(torch.Tensor([[0, - best_rot_angle_y , 0]]))[0,:3,:3].cuda() for best_rot_angle_y in best_rot_angles_y] for best_rot_angles_y in bbest_rot_angles_y]

        bx_d0_crops2 = [[((best_rot @ (x_d0_crop[0] - 0.5).reshape(3,-1)) + 0.5).reshape(*x_d0_crop.shape) for best_rot, x_d0_crop in zip(best_rots, x_d0_crops)] for best_rots, x_d0_crops in zip(bbest_rots,bx_d0_crops)]
        

        # recalculate optimal rotation
        bpred_aligned2scans = []
        for B in range(len(bbest_rots)):
            best_rots = bbest_rots[B]

            if 'pred_scaled_noc2scan_R' in analyses and 'aligned2scan_t' in analyses:
                pred_aligned2scans = []
                for best_rot, pred_R, gt_t, gt_s  in zip(best_rots, analyses['pred_scaled_noc2scan_R'][B], analyses['aligned2scan_t'][B], analyses['aligned2scan_s'][B]):
                    pred_aligned2scan = torch.eye(4).cuda()
                    # Fixing NOC -> Aligned with z flipping
                    pred_aligned2scan[:3,:3] = gt_s * best_rot @ pred_R @  torch.from_numpy(np.diag([1, 1.0, -1])).cuda().float()
                    pred_aligned2scan[:3, 3] = gt_t
                    pred_aligned2scans.append(pred_aligned2scan)
            
            bpred_aligned2scans.append(pred_aligned2scans)
        # debug
        if  False:
            scan_noc_crops = vg_crop(bdscan.scan_noc[0], bbbox_lvl0[0])
            best_rot =  best_rots[0] # angle_axis_to_rotation_matrix(torch.Tensor([[0, np.pi/2, 0]]))[0,:3,:3].cuda()
            gt_rot = ((best_rot.T @ (scan_noc_crops[0][0] - 0.5).view(3,-1)) + 0.5).reshape(*scan_noc_crops[0][0].shape)
            dvis(scan_noc_crops[0][0], add=True, name='gt')
            dvis(gt_rot, add=True, name='gt_rot')

            dvis(x_d0_crops[0][0] * ((scan_noc_crops[0][0][0] > 0).float() - 0.01).float(), add=True, name='pred')
            dvis(x_d0_crops2[0][0]*((scan_noc_crops[0][0][0]>0).float() -0.01).float(),add=True, name='pred_rot')


        return {'noc_values': bx_d0_crops2, 'pred_aligned2scans': bpred_aligned2scans}, losses, analyses, {}
        '''
        bnocs = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)
        losses, analyses, bbest_rot_angles_y = self.loss(bnocs, bbbox_lvl0, bgt_target, bdscan, binst_occ)

        #
        bbest_rots = [[angle_axis_to_rotation_matrix(torch.Tensor([[0, - best_rot_angle_y , 0]]))[0,:3,:3].cuda() for best_rot_angle_y in best_rot_angles_y] for best_rot_angles_y in  bbest_rot_angles_y]

        x_d0_crops2 = [[((best_rot @ (x_d0_crop[0] - 0.5).reshape(3,-1)) + 0.5).reshape(*x_d0_crop.shape) for best_rot, x_d0_crop in zip(best_rots, x_d0_crops)] for best_rots, x_d0_crops in zip(bbest_rots,bnocs)]
        
        # recalculate optimal rotation
        bpred_aligned2scans = []
        for B in range(len(bnocs)):
            pred_aligned2scans = []
            for best_rot, pred_R, pred_t  in zip(bbest_rots[B], analyses['pred_noc2scan_R'][B], analyses['pred_noc2scan_t'][B]):
                pred_noc2scan = torch.eye(4)
                pred_noc2scan[:3,:3] = best_rot @ pred_R
                pred_noc2scan[:3, 3] = pred_t

                pred_aligned2scan = pred_noc2scan @  get_aligned2noc()

                pred_aligned2scans.append(pred_aligned2scan)
            bpred_aligned2scans.append(pred_aligned2scans)

       
        return {'noc_values': x_d0_crops2, 'pred_aligned2scans': bpred_aligned2scans}, losses, analyses, {}

        

    def training_step(self, x_d2: torch.Tensor, x_e2: torch.Tensor, x_e1: torch.Tensor, bdscan, bbbox_lvl0: List, bgt_target: List):
        if self.gt_augm:
            bgt_bbox, bgt_obj_inds = bdscan.bbboxes, bdscan.bobj_inds
            bbbox_lvl0 =  [cats(bbox_lvl0, gt_bbox,0)  for bbox_lvl0, gt_bbox in zip(bbbox_lvl0, bgt_bbox)]   
            bgt_target =  [gt_target + gt_obj_inds  for gt_target, gt_obj_inds in zip(bgt_target,bgt_obj_inds)]
        
        bx_d0_crops = self.forward(x_d2, x_e2, x_e1, bbbox_lvl0)

        losses, analyses, _ = self.loss(bx_d0_crops, bbbox_lvl0, bgt_target, bdscan, None)

        bpred_aligned2scans = []
        for B in range(len(bx_d0_crops)):
            pred_aligned2scans = []
            for pred_R, pred_t  in zip( analyses['pred_noc2scan_R'][B], analyses['pred_noc2scan_t'][B]):
                pred_noc2scan = torch.eye(4)
                pred_noc2scan[:3,:3] = pred_R
                pred_noc2scan[:3, 3] = pred_t

                pred_aligned2scan = pred_noc2scan @  get_aligned2noc()

                pred_aligned2scans.append(pred_aligned2scan)
            bpred_aligned2scans.append(pred_aligned2scans)

        outputs = {'noc_values': bx_d0_crops, 'pred_aligned2scans': bpred_aligned2scans}
        return outputs, losses, analyses, {}

    def rot_sym_loss(self, pred_nocs, gt_nocs, rot_sym, scan_coords=None, noc2scan=None):
        best_rot_angle_y = 0.0
        if rot_sym == 'None':
            noc_loss = self.crit(pred_nocs, gt_nocs)
            
        elif rot_sym == 'c2':
            gt_nocs_180 = (gt_nocs - 0.5)
            gt_nocs_180[:, [0, 2]] *= -1
            gt_nocs_180 += 0.5
            symm_losses = [self.crit(pred_nocs, gt_nocs), self.crit(pred_nocs, gt_nocs_180)]
            minimal_loss_idx = torch.argmin(torch.Tensor(symm_losses))
            if minimal_loss_idx == 1:
                best_rot_angle_y = np.pi
            noc_loss = symm_losses[minimal_loss_idx]
        elif rot_sym == 'c4':
            gt_nocs_90 = (gt_nocs - 0.5)[:, [2, 1, 0]]
            gt_nocs_90[:, 2] *= -1
            gt_nocs_90 += 0.5
            
            gt_nocs_180 = (gt_nocs - 0.5)
            gt_nocs_180[:, [0, 2]] *= -1
            gt_nocs_180 += 0.5

            gt_nocs_270 = (gt_nocs - 0.5)[:, [2, 1, 0]]
            gt_nocs_270[:, 0] *= -1
            gt_nocs_270 += 0.5

            symm_losses = [self.crit(pred_nocs, gt_nocs),
            self.crit(pred_nocs, gt_nocs_90),
            self.crit(pred_nocs, gt_nocs_180),
            self.crit(pred_nocs, gt_nocs_270)]

            minimal_loss_idx = torch.argmin(torch.Tensor(symm_losses))
            noc_loss = symm_losses[minimal_loss_idx]
            best_rot_angle_y = np.pi/2 * minimal_loss_idx.item()

        elif rot_sym == 'inf':
            # TODO CHECK IF FULLY CORRECT
            t_noc2scan_pred, R_noc2scan_pred = self.nocs_to_tr(scan_coords, pred_nocs)
            R_delta = R_noc2scan_pred.T @ noc2scan[:3,:3]/get_scale(noc2scan[:3,:3])

            R_delta_rot_angles = rotation_matrix_to_angle_axis(R_delta.unsqueeze(0))
            R_delta_rot_angles[:,0] = 0
            R_delta_rot_angles[:, 2] = 0
            R_delta_y_only = angle_axis_to_rotation_matrix(R_delta_rot_angles)[0][:3,:3]


            gt_nocs_rotated = ((R_delta_y_only @ (gt_nocs - 0.5).T).T + 0.5)
            
            noc_loss = self.crit(pred_nocs, gt_nocs_rotated)
            best_rot_angle_y = R_delta_rot_angles[:,1]

            # debugging

            # _, R_noc2scan_gt_rot = self.nocs_to_tr(scan_coords, gt_nocs_rotated)
            # R_noc2scan_gt_rot ~ R_noc2scan_pred mod y rot
        elif rot_sym == 'debug':
           

            best_rot_angle_y = 1

            R_delta_y_only = angle_axis_to_rotation_matrix(torch.Tensor([[0,best_rot_angle_y,0]]).cuda())[0][:3,:3]


            gt_nocs_rotated = ((R_delta_y_only @ (gt_nocs - 0.5).T).T + 0.5)
            
            noc_loss = self.crit(pred_nocs, gt_nocs_rotated)
            


        return noc_loss, best_rot_angle_y

    def nocs_to_trs(self, nocs, coords):
        pass
    
    def nocs_to_tr(self,  nocs, coords):
        # returns nocs2coords transform
        indices = np.random.choice(range(len(nocs)), min(len(nocs),self.noc_samples))
        t, R = kabsch_trans_rot(nocs[indices].T, coords[indices].T)
        # +
        return t, R


    def loss(self, bx_d0_crops, bbbox_lvl0: List, bgt_target: List, bdscan, binst_occ=None):

        banalyses = dict()
        bnoc_gt_compl_losses = []
        brot_gt_compl_losses = []
        btransl_gt_compl_losses = []
        bweighted_loss = []

        bbest_rot_angles_y = []

        for B in range(len(bbbox_lvl0)):
            bbox_lvl0 = bbbox_lvl0[B]
            gt_target = bgt_target[B]

            scan_noc_crops = vg_crop(bdscan.bscan_noc[B:B + 1], bbox_lvl0)
            if binst_occ is None:
                scan_inst_mask_crops = vg_crop(bdscan.bscan_inst_mask[B:B+1], bbox_lvl0)
                scan_noc_inst_crops = [((scan_inst_mask_crops[i][:,0] == int(gt_target[i])) & torch.all(scan_noc_crop >=0,1))[0] for i,scan_noc_crop in enumerate(scan_noc_crops)]
            else:
                scan_noc_inst_crops = binst_occ[B]

            analyses = dict()

            # debugging
            if False:
                i= 0 # which bbox
                dvis(bx_d0_crops[B][i][0])
                dvis(scan_noc_crops[i][0], add=True)
            

            noc_gt_compl_losses = []
            rot_gt_compl_losses = []
            transl_gt_compl_losses = []
            best_rot_angles_y = []
            for j in range(len(bbox_lvl0)):
                bbox = bbox_lvl0[j]
                dobject = bdscan.bobjects[B][gt_target[j]]

                inst_occ = scan_noc_inst_crops[j]
                if inst_occ.sum() < 5:
                    bit_mask = np.random.choice([0,1], size=(inst_occ.shape[-3:]))
                    inst_occ[...,bit_mask] = 1 

                scan_noc_inst_crops_grid_coords = torch.nonzero(inst_occ).float()
                pred_noc_on_gt_inst = bx_d0_crops[B][j][0,:, inst_occ].T
                gt_noc_on_gt_inst = scan_noc_crops[j][0,:, inst_occ].T
                # handle object rotational symmetry
                noc_gt_compl_loss_j, best_rot_angle_y = self.rot_sym_loss(pred_noc_on_gt_inst,
                gt_noc_on_gt_inst,
                bdscan.bobjects[B][gt_target[j]].rot_sym,
                scan_noc_inst_crops_grid_coords,
                bdscan.bobjects[B][gt_target[j]].noc2scan)

                # compute rotations
                noc2scan = bdscan.bobjects[B][gt_target[j]].noc2scan
                # using GT scale
                scaled_pred_nocs = pred_noc_on_gt_inst * get_scale(noc2scan)[0]

                pred_scaled_noc2scan_t, pred_scaled_noc2scan_R = self.nocs_to_tr(scaled_pred_nocs, scan_noc_inst_crops_grid_coords + bbox[:3])
                s = torch.diag(get_scale(noc2scan)[:3]).cuda()
                pred_noc2scan_R = pred_scaled_noc2scan_R @ s
                pred_noc2scan_t = pred_scaled_noc2scan_t 

                gt_scaled_noc2scan_R = (noc2scan[:3,:3] / get_scale(noc2scan[:3,:3])) @ angle_axis_to_rotation_matrix(torch.Tensor([[0, -best_rot_angle_y, 0]]))[0,:3,:3].cuda()
                
                # Rotational error
                relative_compl_loss_R = pred_scaled_noc2scan_R @ torch.inverse(gt_scaled_noc2scan_R)
                
                rot_gt_compl_loss_j = torch.norm(relative_compl_loss_R - torch.eye(3).cuda())
                transl_gt_compl_loss_j = torch.norm((pred_noc2scan_t - noc2scan[:3,3])/10)

                delta_rot = rotation_matrix_to_angle_axis(relative_compl_loss_R.unsqueeze(0))[0]
                analyses['rot_angle_diffs'] = analyses.get('rot_angle_diffs', []) + [torch.abs(delta_rot) * 180 / np.pi]
                analyses['transl_diffs'] = analyses.get('transl_diffs', []) + [torch.abs(pred_noc2scan_t - noc2scan[:3,3])]
                analyses['transl_diffs_center'] = analyses.get('transl_diffs_center', []) + [torch.abs( (bbox[:3] + bbox[3:6])/2 - dobject.aligned2scan[:3,3])]
                
                analyses['pred_scaled_noc2scan_R'] = analyses.get('pred_scaled_noc2scan_R', []) + [pred_scaled_noc2scan_R]
                analyses['pred_scaled_noc2scan_t'] = analyses.get('pred_scaled_noc2scan_t', []) + [pred_scaled_noc2scan_t]

                analyses['pred_noc2scan_R'] = analyses.get('pred_noc2scan_R', []) + [pred_noc2scan_R]
                analyses['pred_noc2scan_t'] = analyses.get('pred_noc2scan_t', []) + [pred_noc2scan_t]  
                best_rot_angles_y.append(best_rot_angle_y)

                
                if False:
                    dvis(torch.cat([torch.nonzero(binst_occ).float(), pred_noc_on_gt_inst],1), add=True)
                    dvis(torch.cat([torch.nonzero(binst_occ).float(), gt_noc_on_gt_inst],1), add=True)

                noc_gt_compl_losses.append(noc_gt_compl_loss_j)
                rot_gt_compl_losses.append(rot_gt_compl_loss_j)

                transl_gt_compl_losses.append(transl_gt_compl_loss_j)


            noc_gt_compl_loss = torch.stack(noc_gt_compl_losses)
            rot_gt_compl_loss = torch.stack(rot_gt_compl_losses)
            transl_gt_compl_loss = torch.stack(transl_gt_compl_losses)
            weighted_loss = self.noc_weight *  noc_gt_compl_loss + self.rot_weight * rot_gt_compl_loss + self.transl_weight * transl_gt_compl_loss

            bnoc_gt_compl_losses.append(noc_gt_compl_loss)
            brot_gt_compl_losses.append(rot_gt_compl_loss)
            btransl_gt_compl_losses.append(transl_gt_compl_loss)
            bweighted_loss.append(weighted_loss)

            bbest_rot_angles_y.append(best_rot_angles_y)

            for key in analyses:
                banalyses[key] = banalyses.get(key, []) + [analyses[key]]
        
        losses = {'bweighted_loss': bweighted_loss,
        'bnoc_gt_inst_loss': bnoc_gt_compl_losses,
        'brot_gt_inst_loss': brot_gt_compl_losses,
        'btransl_gt_inst_loss': btransl_gt_compl_losses}
        return losses, banalyses, bbest_rot_angles_y