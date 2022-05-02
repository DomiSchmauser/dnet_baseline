import numpy as np
import motmetrics as mm
import pandas as pd
from dvis import dvis

from utils.net_utils import vg_crop, get_scale

class Tracker:

    def __init__(self):
        self.seq_len = 125
        self.quantization_size = 0.04
        self.similar_value = 0.1
        self.iou_thres = 0.3
        self.l2_thres = 0.4
        self.dist_thres = 100

    def analyse_trajectories(self, gt_seq_df, pred_seq_df, occ_grids):
        '''
        Create trajectories based on match criterion
        '''

        seq_data = dict()

        pred_trajectories = []
        gt_trajectories = []

        for scan_idx in range(self.seq_len):

            occ_grid = occ_grids[scan_idx]

            gt_scan = gt_seq_df.loc[gt_seq_df['scan_idx'] == scan_idx]
            pred_scan = pred_seq_df.loc[pred_seq_df['scan_idx'] == scan_idx]

            gt_scan_dct = gt_scan.to_dict(orient='index')  # idxs, clmn
            pred_scan_dct = pred_scan.to_dict(orient='index')

            shift_free = list(gt_scan_dct.values())[0]['shift'] # world to normalizedworld
            shift_coords = shift_free * (1 / self.quantization_size)
            world2normworld = np.identity(4)
            world2normworld[:3, 3] = shift_free # in original coords #todo check if correct

            cam2world = list(gt_scan_dct.values())[0]['campose'] # Cam2world
            cam_free2world_free = cam2world # Cam2world normed
            normworld2cam = np.linalg.inv(cam2world)

            scan2world = np.identity(4)
            scan2world[:3, :3] = np.diag([self.quantization_size, self.quantization_size, self.quantization_size])
            cam_grid2cam_free = normworld2cam @ world2normworld @ scan2world # Scan2cam

            gt_target = []
            for gt_t in list(gt_scan_dct.values()):
                gt_target.append(gt_t['obj_idx'])

            seq_data[scan_idx] = {'cam_free2world_free': cam_free2world_free, # cam2world
                                  'cam_grid2cam_free': cam_grid2cam_free, # Scan2cam
                                  'gt_target': gt_target}

            # Initialize trajectory
            if scan_idx == 0:
                initial_gt_ids = []
                for obj in pred_scan_dct.values():
                    has_similar = False
                    for pred_traj in pred_trajectories: #todo checks if any close object exists -> some kind of NMS
                        if np.linalg.norm(pred_traj[0]['obj']['pred_aligned2scan'][:3, 3] - obj['pred_aligned2scan'][:3, 3]) < (self.similar_value / self.quantization_size): #todo check this
                            has_similar = True
                    if not has_similar: # not an object which is close #todo added check if gt id already in object -> maybe better
                        if obj['gt_target'] not in initial_gt_ids:
                            pred_trajectories.append([{'obj':obj, 'scan_idx':obj['scan_idx'], 'shift':shift_coords}])
                            initial_gt_ids.append(obj['gt_target'])
                        else:
                            new_rpn_iou = obj['rpn_iou']
                            for drop_idx, pred_obj in enumerate(pred_trajectories):
                                if pred_obj[0]['obj']['gt_target'] == obj['gt_target']:
                                    old_rpn_iou = pred_obj[0]['obj']['rpn_iou']
                                    break
                            if new_rpn_iou > old_rpn_iou:
                                del pred_trajectories[drop_idx]
                                pred_trajectories.append([{'obj': obj, 'scan_idx': obj['scan_idx'], 'shift': shift_coords}])


                for gt_obj in gt_scan_dct.values():
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}])
            else:
                # Match trajectories to initial trajectory
                pred_trajectories = self.pred_trajectory(pred_trajectories, pred_scan_dct, cam_grid2cam_free, occ_grid=occ_grid, traj_crit='with_first', shift=shift_coords, scan_idx=scan_idx)
                # GT Matching
                for gt_obj in gt_scan_dct.values():
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}]) # start new trajectory



        return pred_trajectories, gt_trajectories, seq_data

    def analyse_trajectories_vis(self, gt_seq_df, pred_seq_df, occ_grids):
        '''
        Create trajectories based on match criterion, visualisation setup
        '''

        seq_data = dict()

        pred_trajectories = []
        gt_trajectories = []

        for scan_idx in range(self.seq_len):

            occ_grid = occ_grids[scan_idx]

            gt_scan = gt_seq_df.loc[gt_seq_df['scan_idx'] == scan_idx]
            pred_scan = pred_seq_df.loc[pred_seq_df['scan_idx'] == scan_idx]

            gt_scan_dct = gt_scan.to_dict(orient='index')  # idxs, clmn
            pred_scan_dct = pred_scan.to_dict(orient='index')

            shift_free = list(gt_scan_dct.values())[0]['shift'] # world to normalizedworld
            shift_coords = shift_free * (1 / self.quantization_size)
            world2normworld = np.identity(4)
            world2normworld[:3, 3] = shift_free # in original coords #todo check if correct

            cam2world = list(gt_scan_dct.values())[0]['campose'] # Cam2world
            cam_free2world_free = cam2world # Cam2world normed
            normworld2cam = np.linalg.inv(cam2world)

            scan2world = np.identity(4)
            scan2world[:3, :3] = np.diag([self.quantization_size, self.quantization_size, self.quantization_size])
            cam_grid2cam_free = normworld2cam @ world2normworld @ scan2world # Scan2cam

            gt_target = []
            for gt_t in list(gt_scan_dct.values()):
                gt_target.append(gt_t['obj_idx'])

            seq_data[scan_idx] = {'cam_free2world_free': cam_free2world_free, # cam2world
                                  'cam_grid2cam_free': cam_grid2cam_free, # Scan2cam
                                  'gt_target': gt_target}

            # Initialize trajectory
            if scan_idx == 0:
                initial_gt_ids = []
                for obj in pred_scan_dct.values():
                    has_similar = False
                    for pred_traj in pred_trajectories: #todo checks if any close object exists -> some kind of NMS
                        if np.linalg.norm(pred_traj[0]['obj']['pred_aligned2scan'][:3, 3] - obj['pred_aligned2scan'][:3, 3]) < (self.similar_value / self.quantization_size): #todo check this
                            has_similar = True
                    if not has_similar: # not an object which is close #todo added check if gt id already in object -> maybe better
                        if obj['gt_target'] not in initial_gt_ids:
                            pred_trajectories.append([{'obj':obj, 'scan_idx':obj['scan_idx'], 'shift':shift_coords}])
                            initial_gt_ids.append(obj['gt_target'])
                        else:
                            new_rpn_iou = obj['rpn_iou']
                            for drop_idx, pred_obj in enumerate(pred_trajectories):
                                if pred_obj[0]['obj']['gt_target'] == obj['gt_target']:
                                    old_rpn_iou = pred_obj[0]['obj']['rpn_iou']
                                    break
                            if new_rpn_iou > old_rpn_iou:
                                del pred_trajectories[drop_idx]
                                pred_trajectories.append([{'obj': obj, 'scan_idx': obj['scan_idx'], 'shift': shift_coords}])


                for gt_obj in gt_scan_dct.values():
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}])
            else:
                # Match trajectories to initial trajectory
                pred_trajectories = self.pred_trajectory(pred_trajectories, pred_scan_dct, cam_grid2cam_free, occ_grid=occ_grid, traj_crit='with_first', shift=shift_coords, scan_idx=scan_idx)
                # GT Matching
                for gt_obj in gt_scan_dct.values():
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}]) # start new trajectory



        return pred_trajectories, gt_trajectories, seq_data

    def pred_trajectory(self, trajectories, dscan_j, cam_grid2cam_free, occ_grid=None, traj_crit='with_first', shift=None, scan_idx=None):

        # assign proposals to trajectories based on traj_crit and match_criterion
        traj_prop_matrix = np.zeros((len(trajectories), len(dscan_j.values())))
        iou_prop_matrix = np.zeros((len(trajectories), len(dscan_j.values())))
        prev_prop_matrix = 1000 * np.ones((len(trajectories), len(dscan_j.values())))
        target_list = []
        obj_idx = 0
        for _, obj_j in dscan_j.items():
            # Voxel occupancy comparison
            surf_occ = (vg_crop((occ_grid > 0), obj_j['bbox']) & (obj_j['occ'] > 0))
            #dvis(surf_occ, fmt='voxels')
            #dvis(compl, fmt='voxels')
            obj_j_occ_in_noc = obj_j['noc'][:, surf_occ].T
            obj_j_noc_vg = self.voxelize_unit_pc(obj_j_occ_in_noc)  # voxelized pc

            for traj_idx, traj in enumerate(trajectories):
                start_obj = traj[0]['obj']
                prior_obj = traj[-1]['obj']

                # Voxel occupancy comparison
                surf_occ = (vg_crop((occ_grid > 0), start_obj['bbox']) & (
                            start_obj['occ'] > 0))
                #dvis(surf_occ, fmt='voxels')
                start_obj_j_occ_in_noc = start_obj['noc'][:, surf_occ].T
                start_obj_noc_vg = self.voxelize_unit_pc(start_obj_j_occ_in_noc)

                # Voxel IoU
                iou3d = float((obj_j_noc_vg & start_obj_noc_vg).sum()) / (obj_j_noc_vg | start_obj_noc_vg).sum()
                traj_prop_matrix[traj_idx, obj_idx] = iou3d

                # Box IoU -> first shift to reset scan differences
                prior_box = self.box_to_world(prior_obj['bbox'], traj[-1]['shift'])
                current_box = self.box_to_world(obj_j['bbox'], shift)
                box_iou = self.get_iou_box(current_box, prior_box)
                iou_prop_matrix[traj_idx, obj_idx] = box_iou

                # Alignment
                if traj_crit == 'alignment':
                    prev_prop_matrix[traj_idx, obj_idx] = np.linalg.norm(
                        (cam_grid2cam_free @ prior_obj['pred_aligned2scan'])[:3, 3] - (cam_grid2cam_free @ obj_j[
                            'pred_aligned2scan'])[:3, 3])  # compare cad to cam preds

                # Get GT Target list #todo check all objects of last frame -> if gt id exist in previous trajectory dont start new one with this id
                if (obj_idx+1) == len(dscan_j.values()):
                    target_list.append(prior_obj['gt_target'])


            obj_idx += 1

        # Use max IoU current object with start object to build trajectory
        if traj_crit == 'alignment':
            for obj_id, obj_ious in enumerate(prev_prop_matrix.T): #after T rows objs clmns traj
                idx_miou = np.argmin(obj_ious)
                obj_dict = {'obj': list(dscan_j.values())[obj_id], 'scan_idx': scan_idx, 'shift': shift}
                if obj_ious[idx_miou] < 0.3:
                    if trajectories[idx_miou][-1]['obj']['scan_idx'] != scan_idx: # skip if object with same scan idx already matched
                        trajectories[idx_miou].append(obj_dict)
                else:
                    # If IoU less than 0.3 check overlap occ
                    obj_ious = iou_prop_matrix.T[obj_id]
                    idx_miou = np.argmax(obj_ious)
                    if obj_ious[idx_miou] >= 0.3:
                        if trajectories[idx_miou][-1]['obj']['scan_idx'] != scan_idx:
                            trajectories[idx_miou].append(obj_dict)
                    else:
                        if obj_dict['obj']['gt_target'] not in target_list:
                            trajectories.append([obj_dict])
                            target_list.append(obj_dict['obj']['gt_target'])
        else:
            for obj_id, obj_ious in enumerate(iou_prop_matrix.T): #after T rows objs clmns traj
                idx_miou = np.argmax(obj_ious)
                obj_dict = {'obj': list(dscan_j.values())[obj_id], 'scan_idx': scan_idx, 'shift': shift}
                if obj_ious[idx_miou] >= self.iou_thres:
                    if trajectories[idx_miou][-1]['obj']['scan_idx'] != scan_idx: # skip if object with same scan idx already matched
                        trajectories[idx_miou].append(obj_dict)
                else:
                    # If IoU less than 0.3 check overlap occ
                    traj_occ_ious = traj_prop_matrix.T[obj_id]
                    idx_miou = np.argmax(traj_occ_ious)
                    if traj_occ_ious[idx_miou] >= 0.2: #todo 0.3 good values
                        if trajectories[idx_miou][-1]['obj']['scan_idx'] != scan_idx:
                            trajectories[idx_miou].append(obj_dict)
                    else:
                        if obj_dict['obj']['gt_target'] not in target_list:
                            trajectories.append([obj_dict])
                            target_list.append(obj_dict['obj']['gt_target'])

        return trajectories

    def get_traj_table(self, traj, seq_data, traj_id):
        traj_df = pd.DataFrame()

        for k in range(len(traj)):
            scan_idx = traj[k]['scan_idx']
            if 'gt' in traj_id:
                cad2scan = traj[k]['obj']['aligned2scan']
                aligned2cam_free = seq_data[scan_idx]['cam_grid2cam_free'] @ cad2scan  # Scan2cam # CAD2CAM
            else:
                aligned2cam_free = seq_data[scan_idx]['cam_grid2cam_free'] @ traj[k]['obj']['pred_aligned2scan']

            aligned2world_free = seq_data[scan_idx]['cam_free2world_free'] @ aligned2cam_free # CAD2WORLD
            cam_t = aligned2cam_free[:3, 3]
            world_t = aligned2world_free[:3, 3]

            if 'gt' in traj_id:
                single_df = pd.DataFrame(dict(scan_idx=scan_idx,
                                              traj_id=traj_id,
                                              cam_x=cam_t[0],
                                              cam_y=cam_t[1],
                                              cam_z=cam_t[2],
                                              world_x=world_t[0],
                                              world_y=world_t[1],
                                              world_z=world_t[2],
                                              obj_idx=traj[k]['obj']['obj_idx'] if 'obj_idx' in traj[k]['obj'] else None,
                                              ref_obj_idx=traj[k]['ref_obj_idx'] if 'ref_obj_idx' in traj[k] else None,
                                              gt_obj_idx=[np.array(seq_data[scan_idx]['gt_target'])] if seq_data[scan_idx][
                                                                                                        'gt_target'] is not None else None, # ISSUE CAN HOLD ONLY ONE ID
                                              ), index=[scan_idx]
                                         )
            else:
                single_df = pd.DataFrame(dict(scan_idx=scan_idx,
                                              traj_id=traj_id,
                                              cam_x=cam_t[0],
                                              cam_y=cam_t[1],
                                              cam_z=cam_t[2],
                                              world_x=world_t[0],
                                              world_y=world_t[1],
                                              world_z=world_t[2],
                                              obj_idx=traj[k]['obj']['gt_target'] if 'gt_target' in traj[k][
                                                  'obj'] else None,
                                              ref_obj_idx=traj[k]['ref_obj_idx'] if 'ref_obj_idx' in traj[k] else None,
                                              gt_obj_idx=[np.array(seq_data[scan_idx]['gt_target'])] if
                                              seq_data[scan_idx][
                                                  'gt_target'] is not None else None,  # ISSUE CAN HOLD ONLY ONE ID
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
        l2_th = self.l2_thres

        mh = mm.metrics.create()

        acc = mm.MOTAccumulator(auto_id=True)
        for scan_idx in range(self.seq_len):
            #gt_cams = np.array(mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx][['cam_x', 'cam_y', 'cam_z']]) # CAD2WORLD TRANSLATION
            gt_cams = np.array(mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx][['world_x', 'world_y', 'world_z']]) # CAD2WORLD TRANSLATION
            # get gt position in camera frame
            gt_objects = mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx]['obj_idx'].tolist()

            hypo_table = pred_table[(pred_table['scan_idx'] == scan_idx)]
            pred_objects = []
            dist_matrix = np.nan * np.ones((len(gt_objects), len(hypo_table)))
            for j, hypo in enumerate(hypo_table.iterrows()):
                #hypo_cam = np.array(hypo[1][['cam_x', 'cam_y', 'cam_z']]) #CAD2WORLD TRANSLATION
                hypo_cam = np.array(hypo[1][['world_x', 'world_y', 'world_z']]) #CAD2WORLD TRANSLATION
                # get hypo position in camera frame
                hypo_id = hypo[1]['obj_idx']
                pred_objects.append(hypo_id)
                for i, gt_obj in enumerate(gt_objects):
                    gt_cam = gt_cams[i,:]
                    dist_matrix[i][j] = mm.distances.norm2squared_matrix(gt_cam, hypo_cam, max_d2=l2_th) # l2 distance between gt object and hypothesis, capped to l2_th


            acc.update(
                gt_objects,  # Ground truth objects in this frame
                pred_objects,  # Detector hypotheses in this frame
                dist_matrix
            )

        all_traj_summary = mh.compute(acc, metrics=['num_frames', 'mota', 'precision', 'recall', 'num_objects', 'num_matches', 'num_misses',
                                           'num_false_positives', 'num_switches'], name='acc')

        return all_traj_summary

    def voxelize_unit_pc(self, pc, shape=20):
        indices = (np.clip(pc, 0, 0.9999) * shape).astype(int)
        vg = np.zeros([shape, shape, shape]).astype(bool)
        vg[tuple(indices.T)] = True
        return vg

    def get_iou_box(self, boxA, boxB):
        """ Compute IoU of two bounding boxes.
        """
        # determine the (x, y, z)-coordinates of the intersection rectangle
        minx_overlap = max(boxA[0], boxB[0])
        miny_overlap = max(boxA[1], boxB[1])
        minz_overlap = max(boxA[2], boxB[2])

        maxx_overlap = min(boxA[3], boxB[3])
        maxy_overlap = min(boxA[4], boxB[4])
        maxz_overlap = min(boxA[5], boxB[5])

        # compute the area of intersection rectangle
        interArea = max(0, maxx_overlap - minx_overlap)*max(0, maxy_overlap - miny_overlap)*max(0, maxz_overlap - minz_overlap)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[3] - boxA[0])*(boxA[4] - boxA[1])*(boxA[5] - boxA[2])
        boxBArea = (boxB[3] - boxB[0])*(boxB[4] - boxB[1])*(boxB[5] - boxB[2])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def box_to_world(self, box, shift):
        """ Reset initial pc shift
        """
        copy_box = box.copy()
        assert copy_box[0] < 192 and copy_box[1] < 192 and copy_box[2] < 96

        # Clip boxes to scan size
        copy_box[3] = np.clip(copy_box[3], 0, 191)
        copy_box[4] = np.clip(copy_box[4], 0, 191)
        copy_box[5] = np.clip(copy_box[5], 0, 95)

        # Reset shift
        copy_box[0] += shift[0] #x
        copy_box[3] += shift[0]
        copy_box[1] += shift[1] #y
        copy_box[4] += shift[1]
        copy_box[2] += shift[2] #z
        copy_box[5] += shift[2]

        return copy_box

    '''
  TRAJECTORY WISE MATCHING
  for traj_id, traj_ious in enumerate(iou_prop_matrix):
      idx_miou = np.argmax(traj_ious)
      obj_dict = {'obj': list(dscan_j.values())[idx_miou], 'scan_idx': scan_idx, 'shift':shift}

      #todo check if scan id exist in trajectory
      if traj_ious[idx_miou] >= self.iou_thres:
          trajectories[traj_id].append(obj_dict)
      else:
          # If IoU less than 0.3 check overlap occ
          traj_occ_ious = traj_prop_matrix[traj_id]
          idx_miou = np.argmax(traj_occ_ious)
          if traj_occ_ious[idx_miou] >= 0.0:
              trajectories[traj_id].append(obj_dict)
          else:
              trajectories.append([obj_dict])
    
    GT TRAJ MATCHING 
    for gt_traj in gt_trajectories:
        matched = False 
        for gt_obj in gt_scan_dct.values():
            if gt_traj[0]['obj']['obj_idx'] == gt_obj['obj_idx']:
                gt_traj.append({'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}) # build list per trajectory
                matched = True
                break

        if not matched:
            gt_trajectories.append([{'obj': gt_obj, 'scan_idx': gt_obj['scan_idx']}]) # start new trajectory #todo check if works
    '''