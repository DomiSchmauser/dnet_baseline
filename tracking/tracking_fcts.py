import numpy as np
import json
from trimesh.transformations import translation_matrix, rotation_matrix, scale_and_translate, scale_matrix, reflection_matrix
import motmetrics as mm
import pandas as pd



#### helper

def get_moving_obj(gt_dscan_i):
    # single moving object in validation sequences
    mov_idx = -1
    for  obj_idx, obj in gt_dscan_i.objects.items():
        if  obj.is_static:
            mov_idx = obj_idx
            break
    
    if mov_idx not in gt_dscan_i.objects:
        return None
    
    gt_obj_i = gt_dscan_i.objects[mov_idx]
    return gt_obj_i


def voxelize_unit_pc(pc, shape=20):
    indices = (np.clip(pc,0,0.9999) * shape).astype(int)
    vg = np.zeros([shape, shape, shape]).astype(bool)
    vg[tuple(indices.T)] = True
    return vg



def vg_crop(vg, bboxes, spatial_end=True, crop_box=False):
    # vg: ... X W X L X H
    # bboxes: N x (min, max,...) or (min,max,...)
    if len(bboxes.shape) == 1:
        if spatial_end:
            if not crop_box:
                assert np.all(bboxes[:3] >= 0) and np.all(bboxes[3:6] < vg.shape[-3:])
                return vg[..., int(bboxes[0]) : int(bboxes[3]), int(bboxes[1]) : int(bboxes[4]), int(bboxes[2]) : int(bboxes[5])]
            else:
                bbox_cropped = np.concatenate([np.max([bboxes[:3], np.zeros(3)], 0), np.min([bboxes[3:], vg.shape[-3:]], 0)], 0)
                return vg[
                    ...,
                    int(bbox_cropped[0]) : int(bbox_cropped[3]),
                    int(bbox_cropped[1]) : int(bbox_cropped[4]),
                    int(bbox_cropped[2]) : int(bbox_cropped[5]),
                ]
        else:
            if not crop_box:
                assert np.all(bboxes[:3] >= 0) and np.all(bboxes[3:6] < vg.shape[:3])
                return vg[int(bboxes[0]) : int(bboxes[3]), int(bboxes[1]) : int(bboxes[4]), int(bboxes[2]) : int(bboxes[5])]
            else:
                bbox_cropped = np.concatenate([np.max([bboxes[:3], np.zeros(3)], 0), np.min([bboxes[3:], vg.shape[:3]], 0)], 0)
                return vg[
                    int(bbox_cropped[0]) : int(bbox_cropped[3]),
                    int(bbox_cropped[1]) : int(bbox_cropped[4]),
                    int(bbox_cropped[2]) : int(bbox_cropped[5]),
                ]
    if len(bboxes.shape) == 2:
        return [vg_crop(vg, bbox, spatial_end, crop_box) for bbox in bboxes]

def dot(transform, points):
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

#####

def pred_trajectory(trajectories, dscan_j, cam_grid2cam_free, obj0, traj_crit='with_first', match_criterion='iou0'):
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
                        obj_j_occ_in_noc = dot(np.linalg.inv(obj_j.noc2bbox_amodal), np.argwhere(obj_j.occ)) # bbox to nocs
                    else:
                        if match_criterion == 'iou0_segm':
                            surf_occ = (vg_crop((np.abs(dscan_j.tsdf_geo)<0.4), obj_j.bbox)  & (obj_j.occ > 0))
                            obj_j_occ_in_noc = obj_j.noc[:, surf_occ].T
                        else:
                            obj_j_occ_in_noc = obj_j.noc[:, obj_j.occ > 0].T
                    obj_j.occ_in_noc = obj_j_occ_in_noc

                obj_j_noc_vg =  voxelize_unit_pc(obj_j.occ_in_noc) # voxelized pc
                best_traj_idx, best_iou = -1, 0
                for traj_idx, traj in enumerate(trajectories):
                    start_obj = traj[0]['obj']
                    
                    if not hasattr(start_obj, 'occ_in_noc'):
                        if obj_j.sdf is not None:
                            # is gt
                            start_obj.occ = start_obj.sdf < 2
                            start_obj_j_occ_in_noc = dot(np.linalg.inv(start_obj.noc2bbox_amodal), np.argwhere(start_obj.occ))
                        else:
                            if match_criterion == 'iou0_segm':
                                surf_occ = (vg_crop((np.abs(dscan_j.tsdf_geo)<0.4), start_obj.bbox)  & (start_obj.occ > 0))
                                start_obj_j_occ_in_noc = start_obj.noc[:, surf_occ].T
                            else:
                                start_obj_j_occ_in_noc = start_obj.noc[:, start_obj.occ > 0].T
                            
                        start_obj.occ_in_noc = start_obj_j_occ_in_noc

                    start_obj_noc_vg = voxelize_unit_pc(start_obj.occ_in_noc)

                    iou3d = float((obj_j_noc_vg & start_obj_noc_vg).sum()) / ( obj_j_noc_vg | start_obj_noc_vg).sum()
                    traj_prop_matrix[traj_idx, obj_idx] = iou3d

                    
                    if int(dscan_j.scan_idx) - int(traj[-1]['scan_idx']) < 10:
                        # last hypo of this trajectory is tempory close
                        prev_obj = traj[-1]['obj']
                        prev_prop_matrix[traj_idx, obj_idx] = np.linalg.norm( (cam_grid2cam_free @ prev_obj.aligned2scan)[:3,3]-  (cam_grid2cam_free @ obj_j.aligned2scan)[:3,3])


    return trajectories


def analyse_trajectories(seq_name, match_criterion ):
     # compute pred/gt trajectories based on match_criterion

    # ugly way to get the length of the sequence
    gt_dseq = DSeq.load_from_file(seq_name, seq_name.name, None, [], [], True, [])
    scan_inds = [str(x) for x in sorted([int(x) for x in list(gt_dseq.scans.keys())])]

    seq_data = dict()

    pred_trajectories = []
    gt_trajectories = []

    for i in range(len(scan_inds)):
        scan_idx = scan_inds[i]

        gt_dseq = DSeq.load_from_file(seq_name, seq_name.name, [scan_idx],[])
        gt_dscan_i = list(gt_dseq.scans.values())[0]

        pred_dseq = DSeq.load_from_file(seq_name, 'pred_%s'%seq_name.name, [scan_idx]) #,['tsdf_geo', 'tsdf_col'] if with_vis else [])
        pred_dscan_i = list(pred_dseq.scans.values())[0]

        cam_free2world_free = np.array(gt_dscan_i.camera_pose) @ reflection_matrix([0, 0, 0], [0, 0, 1]) # Cam2world
        cam_grid2cam_free = np.linalg.inv(cam_free2world_free) @ gt_dscan_i.scan2world # maybe discretized to free

        gt_target= get_moving_obj(gt_dscan_i)
        seq_data[scan_idx] = {'cam_free2world_free': cam_free2world_free,
        'cam_grid2cam_free': cam_grid2cam_free,
        'gt_target': gt_target}
        pred_dscan_i.tsdf_geo = gt_dscan_i.tsdf_geo

        # Initialize trajectory
        if i == 0:
            for obj in pred_dscan_i.objects.values():
                has_similar = False
                for pred_traj in pred_trajectories: # Cad to scan
                    if np.linalg.norm(pred_traj[0]['obj'].aligned2scan[:3, 3] - obj.aligned2scan[:3, 3]) < 0.6 / 0.03: # 0.6m / 0.03 = quantization size?
                        has_similar = True
                if not has_similar: # ???
                    pred_trajectories.append([{'obj':obj, 'scan_idx':pred_dscan_i.scan_idx}]) # Initial objects which dont match with GT objects start trajectory ??
            for obj in gt_dscan_i.objects.values():
                gt_trajectories.append([{'obj': obj, 'scan_idx': gt_dscan_i.scan_idx}]) # All initial objects
        else:
            # Match trajectories to initial trajectory
            pred_trajectories = pred_trajectory(pred_trajectories, pred_dscan_i, cam_grid2cam_free, None, traj_crit='with_first', match_criterion=match_criterion)
            for gt_traj in gt_trajectories:
                for gt_obj in gt_dscan_i.objects.values():
                    if gt_traj[0]['obj'].obj_idx == gt_obj.obj_idx:
                        gt_traj.append({'obj': gt_obj, 'scan_idx': gt_dscan_i.scan_idx}) # build list per trajectory
                        break
            
    return pred_trajectories, gt_trajectories, seq_data




def eval_mota(pred_table, mov_obj_traj_table):
    # compute mota based on l2_th
    l2_th = 0.25   

    mh = mm.metrics.create()
    all_traj_summary = pd.DataFrame()

    for pred_traj_id in pred_table['traj_id'].drop_duplicates():
        acc = mm.MOTAccumulator(auto_id=True)
        for scan_idx in mov_obj_traj_table['scan_idx']:
                gt_cam = np.array(mov_obj_traj_table[mov_obj_traj_table['scan_idx']==scan_idx][['cam_x','cam_y', 'cam_z']])
                # get gt position in camera frame
                gt_idx = int(mov_obj_traj_table[mov_obj_traj_table['scan_idx']==scan_idx]['obj_idx'])
                
                gt_objects = [gt_idx]
                
                hypo_table = pred_table[(pred_table['scan_idx']==scan_idx) & (pred_table['traj_id']==pred_traj_id)]
                pred_objects = []
                dist_matrix = np.nan*np.ones((len(gt_objects), len(hypo_table)))
                #print(dist_matrix.shape)
                for j,hypo in enumerate(hypo_table.iterrows()):
                    hypo_cam = np.array(hypo[1][['cam_x','cam_y', 'cam_z']])
                    # get hypo position in camera frame
                    hypo_id = int(hypo[1]['traj_id'].split('_')[-1]) # format was pred_X
                    pred_objects.append(hypo_id)
                    for i,gt_obj in enumerate(gt_objects):
                        # SINGLE GT OBJECT THOUGH
                        dist_matrix[i][j] = mm.distances.norm2squared_matrix(gt_cam, hypo_cam, max_d2=l2_th)
                        # l2 distance between gt object and hypothesis, capped to l2_th
                
                acc.update(
                    gt_objects,                     # Ground truth objects in this frame
                    pred_objects,                  # Detector hypotheses in this frame
                    dist_matrix
                )
        
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_matches', 'num_misses', 'num_false_positives'], name='acc')
        summary['traj_id'] = pred_traj_id
        all_traj_summary = pd.concat([all_traj_summary, summary])
    return all_traj_summary


def get_traj_table(traj, seq_data, traj_id):
    traj_df = pd.DataFrame()
    
    for k in range(len(traj)):
        scan_idx = traj[k]['scan_idx']
        aligned2cam_free = seq_data[scan_idx]['cam_grid2cam_free']   @ traj[k]['obj'].aligned2scan
        aligned2world_free = seq_data[scan_idx]['cam_free2world_free'] @aligned2cam_free
        cam_t = aligned2cam_free[:3,3]
        world_t = aligned2world_free[:3,3]

        single_df = pd.DataFrame(dict(scan_idx=scan_idx,
        traj_id=traj_id,
        cam_x=cam_t[0],
        cam_y=cam_t[1],
        cam_z=cam_t[2],
        world_x=world_t[0],
        world_y=world_t[1],
        world_z=world_t[2],
        obj_idx=traj[k]['obj'].obj_idx,
        ref_obj_idx=traj[k].ref_obj_idx if hasattr(traj[k], 'ref_obj_idx') else None,
        gt_obj_idx=seq_data[scan_idx]['gt_target'].obj_idx if seq_data[scan_idx]['gt_target'] is not None else None,
        ), index=[scan_idx]
        )

        traj_df = pd.concat([traj_df, single_df],axis=0)
    return traj_df

def get_traj_tables(trajectories, seq_data, prefix):
    traj_tables = pd.DataFrame()
    for t in range(len(trajectories)):
        traj_table = get_traj_table(trajectories[t], seq_data, f'{prefix}_{t}')
        traj_tables = pd.concat([traj_tables, traj_table], axis=0)
    return traj_tables



if __name__ == "__main__":
    crit = 'iou0_segm'
    seq_name = "???"
    pred_trajectories, gt_trajectories, seq_data = analyse_trajectories(seq_name, crit)
    pred_traj_tables = get_traj_tables(pred_trajectories, seq_data, 'pred')
    gt_traj_tables = get_traj_tables(gt_trajectories, seq_data, 'gt')
    pred_traj_tables.to_csv(f'pred_traj_{seq_name.name}_segm.csv', index=False)
    gt_traj_tables.to_csv(f'gt_traj_{seq_name.name}_segm.csv', index=False)
