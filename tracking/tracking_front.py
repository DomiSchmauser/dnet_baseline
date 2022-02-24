import numpy as np
import json
from trimesh.transformations import translation_matrix, rotation_matrix, scale_and_translate, scale_matrix, reflection_matrix
import motmetrics as mm
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

            gt_scan_dct = gt_scan.to_dict(orient='index')  # idxs, clmn
            pred_scan_dct = pred_scan.to_dict(orient='index')

            cam_free2world_free = list(gt_scan_dct.values())[0]['campose'] @ reflection_matrix([0, 0, 0], [0, 0, 1]) # Cam2world
            cam_grid2cam_free = np.linalg.inv(cam_free2world_free) #@ gt_dscan_i.scan2world # maybe discretized to free
            cam_grid2cam_free[:3,3] *= 0.04 #transfrom from discretized to coords

            gt_target = 0
            seq_data[scan_idx] = {'cam_free2world_free': cam_free2world_free,
                                  'cam_grid2cam_free': cam_grid2cam_free,
                                  'gt_target': gt_target}

            # Initialize trajectory
            if scan_idx == 0:
                for obj in pred_scan_dct.values():
                    has_similar = False
                    for pred_traj in pred_trajectories: # Cad to scan
                        if np.linalg.norm(pred_traj[0]['obj']['pred_aligned2scan'][:3, 3] - obj['pred_aligned2scan'][:3, 3]) < 0.6 / 0.04: # 0.6m / 0.03 = quantization size?
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
                        if gt_traj[0]['obj']['obj_idx'] == gt_obj['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}) # build list per trajectory
                            break

        return pred_trajectories, gt_trajectories, seq_data

    def pred_trajectory(self, trajectories, dscan_j, cam_grid2cam_free, obj0, occ_grid=None, traj_crit='with_first', match_criterion='iou0'):

        # assign proposals to trajectories based on traj_crit and match_criterion
        traj_prop_matrix = np.zeros((len(trajectories), len(dscan_j.values())))
        prev_prop_matrix = 1000 * np.ones((len(trajectories), len(dscan_j.values())))
        obj_idx = 0
        for _, obj_j in dscan_j.items():
            if traj_crit == 'with_first':
                if match_criterion == 'iou0' or match_criterion == 'iou0_segm':
                    surf_occ = (vg_crop((occ_grid > 0), obj_j['bbox']) & (obj_j['occ'] > 0))
                    obj_j_occ_in_noc = obj_j['noc'][:, surf_occ].T

                    obj_j['occ_in_noc'] = obj_j_occ_in_noc
                    obj_j_noc_vg = self.voxelize_unit_pc(obj_j_occ_in_noc)  # voxelized pc

                    for traj_idx, traj in enumerate(trajectories):
                        start_obj = traj[0]['obj']

                        if match_criterion == 'iou0':
                            surf_occ = (vg_crop((occ_grid > 0), start_obj['bbox']) & (
                                        start_obj['occ'] > 0))
                            start_obj_j_occ_in_noc = start_obj['noc'][:, surf_occ].T

                            start_obj['occ_in_noc'] = start_obj_j_occ_in_noc

                        start_obj_noc_vg = self.voxelize_unit_pc(start_obj_j_occ_in_noc)

                        iou3d = float((obj_j_noc_vg & start_obj_noc_vg).sum()) / (obj_j_noc_vg | start_obj_noc_vg).sum()
                        traj_prop_matrix[traj_idx, obj_idx] = iou3d

                        if int(list(dscan_j.values())[0]['scan_idx']) - int(traj[-1]['scan_idx']) < 10:
                            # last hypo of this trajectory is tempory close
                            prev_obj = traj[-1]['obj']
                            prev_prop_matrix[traj_idx, obj_idx] = np.linalg.norm(
                                (cam_grid2cam_free @ prev_obj['pred_aligned2scan'])[:3, 3] - (cam_grid2cam_free @ obj_j['pred_aligned2scan'])[:3, 3])

            obj_idx += 1
        return trajectories

    def get_traj_table(self, traj, seq_data, traj_id):
        traj_df = pd.DataFrame()

        for k in range(len(traj)):
            scan_idx = traj[k]['scan_idx']
            if 'gt' in traj_id:
                aligned2cam_free = seq_data[scan_idx]['cam_grid2cam_free'] @ traj[k]['obj']['aligned2scan']
            else:
                aligned2cam_free = seq_data[scan_idx]['cam_grid2cam_free'] @ traj[k]['obj']['pred_aligned2scan']

            aligned2world_free = seq_data[scan_idx]['cam_free2world_free'] @ aligned2cam_free
            cam_t = aligned2cam_free[:3, 3]
            world_t = aligned2world_free[:3, 3]

            single_df = pd.DataFrame(dict(scan_idx=scan_idx,
                                          traj_id=traj_id,
                                          cam_x=cam_t[0],
                                          cam_y=cam_t[1],
                                          cam_z=cam_t[2],
                                          world_x=world_t[0],
                                          world_y=world_t[1],
                                          world_z=world_t[2],
                                          obj_idx=traj[k]['obj']['obj_idx'],
                                          ref_obj_idx=traj[k]['ref_obj_idx'] if 'ref_obj_idx' in traj[k] else None,
                                          gt_obj_idx=seq_data[scan_idx]['gt_target'] if seq_data[scan_idx][
                                                                                                    'gt_target'] is not None else None,
                                          ), index=[scan_idx]
                                     )

            traj_df = pd.concat([traj_df, single_df], axis=0)
        return traj_df

    def get_traj_tables(self, trajectories, seq_data, prefix):
        traj_tables = pd.DataFrame()
        for t in range(len(trajectories)):
            traj_table = self.get_traj_table(trajectories[t], seq_data, f'{prefix}_{t}')
            traj_tables = pd.concat([traj_tables, traj_table], axis=0)
        return traj_tables

    def eval_mota(self, pred_table, mov_obj_traj_table):
        # compute mota based on l2_th
        l2_th = 0.25

        mh = mm.metrics.create()
        all_traj_summary = pd.DataFrame()

        for pred_traj_id in pred_table['traj_id'].drop_duplicates():
            acc = mm.MOTAccumulator(auto_id=True)
            for scan_idx in mov_obj_traj_table['scan_idx']:
                gt_cam = np.array(
                    mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx][['cam_x', 'cam_y', 'cam_z']])
                # get gt position in camera frame
                gt_idx = int(mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx]['obj_idx'])

                gt_objects = [gt_idx]

                hypo_table = pred_table[(pred_table['scan_idx'] == scan_idx) & (pred_table['traj_id'] == pred_traj_id)]
                pred_objects = []
                dist_matrix = np.nan * np.ones((len(gt_objects), len(hypo_table)))
                # print(dist_matrix.shape)
                for j, hypo in enumerate(hypo_table.iterrows()):
                    hypo_cam = np.array(hypo[1][['cam_x', 'cam_y', 'cam_z']])
                    # get hypo position in camera frame
                    hypo_id = int(hypo[1]['traj_id'].split('_')[-1])  # format was pred_X
                    pred_objects.append(hypo_id)
                    for i, gt_obj in enumerate(gt_objects):
                        # SINGLE GT OBJECT THOUGH
                        dist_matrix[i][j] = mm.distances.norm2squared_matrix(gt_cam, hypo_cam, max_d2=l2_th)
                        # l2 distance between gt object and hypothesis, capped to l2_th

                acc.update(
                    gt_objects,  # Ground truth objects in this frame
                    pred_objects,  # Detector hypotheses in this frame
                    dist_matrix
                )

            summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_matches', 'num_misses',
                                               'num_false_positives'], name='acc')
            summary['traj_id'] = pred_traj_id
            all_traj_summary = pd.concat([all_traj_summary, summary])
        return all_traj_summary

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
