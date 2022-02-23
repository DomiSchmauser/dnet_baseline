import numpy as np
import json
from trimesh.transformations import translation_matrix, rotation_matrix, scale_and_translate, scale_matrix, reflection_matrix
#import motmetrics as mm
import pandas as pd

from utils.net_utils import vg_crop

class Tracker:

    def __init__(self):
        self.seq_len = 25
        self.match_criterion = 'iou0'

    def analyse_trajectories(self, gt_seq_df, pred_seq_df, occ_grid):
        '''
        Create trajectories based on match criterion
        '''

        seq_data = dict()

        pred_trajectories = []
        gt_trajectories = []

        for scan_idx in range(self.seq_len):

            gt_scan = gt_seq_df.loc[gt_seq_df['scan_idx'] == scan_idx]
            pred_scan = pred_seq_df.loc[pred_seq_df['scan_idx'] == scan_idx]

            cam_free2world_free = gt_scan.loc[0, 'campose'] @ reflection_matrix([0, 0, 0], [0, 0, 1]) # Cam2world
            cam_grid2cam_free = np.linalg.inv(cam_free2world_free) #@ gt_dscan_i.scan2world # maybe discretized to free

            gt_target = 0
            seq_data[scan_idx] = {'cam_free2world_free': cam_free2world_free,
                                  'cam_grid2cam_free': cam_grid2cam_free,
                                  'gt_target': gt_target}

            gt_scan_dct = gt_scan.to_dict(orient='index') #idxs, clmn
            pred_scan_dct = pred_scan.to_dict(orient='index')
            #pred_dscan_i.tsdf_geo = gt_dscan_i.tsdf_geo

            # Initialize trajectory
            if scan_idx == 0:
                for obj in pred_scan_dct.values():
                    has_similar = False
                    for pred_traj in pred_trajectories: # Cad to scan
                        if np.linalg.norm(pred_traj[0]['obj'].aligned2scan[:3, 3] - obj['aligned2scan'][:3, 3]) < 0.6 / 0.04: # 0.6m / 0.03 = quantization size?
                            has_similar = True
                    if not has_similar: # not an object which is close
                        pred_trajectories.append([{'obj':obj, 'scan_idx':obj['scan_idx']}])
                for obj in gt_scan_dct.values():
                    gt_trajectories.append([{'obj': obj, 'scan_idx': obj['scan_idx']}]) # All initial objects
            else:
                # Match trajectories to initial trajectory
                pred_trajectories = self.pred_trajectory(pred_trajectories, pred_scan_dct, cam_grid2cam_free, None, occ_grid=occ_grid, traj_crit='with_first', match_criterion=self.match_criterion)
                for gt_traj in gt_trajectories:
                    for gt_obj in gt_scan_dct.values():
                        if gt_traj[0]['obj'].obj_idx == gt_obj.obj_idx:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}) # build list per trajectory
                            break

        return pred_trajectories, gt_trajectories, seq_data

    def pred_trajectory(self, trajectories, dscan_j, cam_grid2cam_free, obj0, occ_grid=None, traj_crit='with_first', match_criterion='iou0'):
        # assign proposals to trajectories based on traj_crit and match_criterion
        traj_prop_matrix = np.zeros((len(trajectories), len(dscan_j.objects)))
        prev_prop_matrix = 1000 * np.ones((len(trajectories), len(dscan_j.objects)))
        proposal_ids = list(dscan_j.objects.keys())
        for obj_idx, (obj_id_j, obj_j) in enumerate(dscan_j.objects.items()):
            if traj_crit == 'with_first':
                if match_criterion == 'iou0' or match_criterion == 'iou0_segm':
                    if not hasattr(obj_j, 'obj_j_occ_in_noc'):
                        if obj_j.sdf is not None:
                            # is gt
                            obj_j.occ = obj_j.sdf < 2
                            obj_j_occ_in_noc = self.dot(np.linalg.inv(obj_j.noc2bbox_amodal),
                                                   np.argwhere(obj_j.occ))  # bbox to nocs
                        else:
                            if match_criterion == 'iou0_segm':
                                surf_occ = (vg_crop((occ_grid > 0), obj_j.bbox) & (obj_j.occ > 0))
                                obj_j_occ_in_noc = obj_j.noc[:, surf_occ].T
                            else:
                                obj_j_occ_in_noc = obj_j.noc[:, obj_j.occ > 0].T
                        obj_j.occ_in_noc = obj_j_occ_in_noc

                    obj_j_noc_vg = self.voxelize_unit_pc(obj_j.occ_in_noc)  # voxelized pc
                    best_traj_idx, best_iou = -1, 0
                    for traj_idx, traj in enumerate(trajectories):
                        start_obj = traj[0]['obj']

                        if not hasattr(start_obj, 'occ_in_noc'):
                            if obj_j.sdf is not None:
                                # is gt
                                start_obj.occ = start_obj.sdf < 2
                                start_obj_j_occ_in_noc = self.dot(np.linalg.inv(start_obj.noc2bbox_amodal),
                                                             np.argwhere(start_obj.occ))
                            else:
                                if match_criterion == 'iou0_segm':
                                    surf_occ = (vg_crop((occ_grid > 0), start_obj.bbox) & (
                                                start_obj.occ > 0))
                                    start_obj_j_occ_in_noc = start_obj.noc[:, surf_occ].T
                                else:
                                    start_obj_j_occ_in_noc = start_obj.noc[:, start_obj.occ > 0].T

                            start_obj.occ_in_noc = start_obj_j_occ_in_noc

                        start_obj_noc_vg = self.voxelize_unit_pc(start_obj.occ_in_noc)

                        iou3d = float((obj_j_noc_vg & start_obj_noc_vg).sum()) / (obj_j_noc_vg | start_obj_noc_vg).sum()
                        traj_prop_matrix[traj_idx, obj_idx] = iou3d

                        if int(dscan_j.scan_idx) - int(traj[-1]['scan_idx']) < 10:
                            # last hypo of this trajectory is tempory close
                            prev_obj = traj[-1]['obj']
                            prev_prop_matrix[traj_idx, obj_idx] = np.linalg.norm(
                                (cam_grid2cam_free @ prev_obj.aligned2scan)[:3, 3] - (cam_grid2cam_free @ obj_j.aligned2scan)[:3, 3])

        return trajectories

    def dot(self, transform, points):
        if type(points) == list:
            points = np.array(points)

        if len(points.shape) == 1:
            # single point
            if transform.shape == (3, 3):
                return transform @ points[:3]
            else:
                return (transform @ np.array([*points[:3], 1]))[:3]
        elif len(points.shape) == 2:

            if points.shape[0] not in [3, 4] and points.shape[1] in [3, 4]:
                # needs to be transposed for dot product
                points = points.T
        else:
            raise RuntimeError("Format of points not understood")
        # points in format [3/4,n]
        if transform.shape == (4, 4):
            return (transform[:3, :3] @ points[:3]).T + transform[:3, 3]
        elif transform.shape == (3, 3):
            return (transform[:3, :3] @ points[:3]).T
        else:
            raise RuntimeError("Format of transform not understood")

    def voxelize_unit_pc(self, pc, shape=20):
        indices = (np.clip(pc, 0, 0.9999) * shape).astype(int)
        vg = np.zeros([shape, shape, shape]).astype(bool)
        vg[tuple(indices.T)] = True
        return vg

