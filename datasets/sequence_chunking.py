import torch
import numpy as np
import MinkowskiEngine as ME
from torch.nn import functional as F


def chunk_sequence(seq, chunk_size=1, device=None):
    '''
    Split sequence with 25 images in chunks to fit memory
    returns:
    dense_features: list of n chunks with batched features shape: BS(num imgs) x dims
    sparse_features: list of n chunks with batched sparse tensors
    sparse_reg_features: list of per image features, len=25
    '''

    occ = []
    coords = []
    feats = []
    sparse_features = []
    dense_features = []
    sparse_reg_features = []
    sparse_obj_features = []
    sparse_box_features = []
    bscan_inst_mask = []
    bscan_obj = []

    for idx, img in enumerate(seq):

        # Sequence Level
        img_grid = torch.unsqueeze(img['dense_grid'], dim=0)
        occ.append(img_grid)
        coords.append(img['sparse_coords'])
        feats.append(img['sparse_feats'])

        # Object level
        reg_values = np.zeros([1, 7, img['dense_grid'].shape[0], img['dense_grid'].shape[1], img['dense_grid'].shape[2]])
        scan_inst_mask = img['obj_scan_mask']
        bboxes = []
        obj_idxs = []
        obj_feats = {}

        for obj in img['obj_anns']:

            box_3d = obj['box_3d']
            obj_idx = obj['instance_id']

            obj_scan_mask = scan_inst_mask == int(obj_idx)
            obj_scan_mask = obj_scan_mask.numpy()
            #Todo add assert obj idx only once in dict
            obj_feats[str(obj_idx)] = {}
            obj_feats[str(obj_idx)]['num_occ'] = obj_scan_mask.sum()# counts True elements in scan
            obj_feats[str(obj_idx)]['noc2scan'] = obj['rot']
            obj_feats[str(obj_idx)]['rot_sym'] =  obj['rot_sym']

            '''
            # NOT SURE IF NECESSARY
            obj_scan_mask_torch = obj_scan_mask.cuda()
            obj_scan_mask_torch2 = F.conv3d(
                obj_scan_mask_torch.float().unsqueeze(0).unsqueeze(0), weight=torch.ones(1, 1, 9, 9, 9).cuda(),
                padding=4
            )[0][0]
            obj_scan_mask = (obj_scan_mask_torch2 > 0).cpu().numpy()
            '''

            obj_scan_coords = np.argwhere(obj_scan_mask)

            obj_center = (box_3d[3:6] + box_3d[:3]) / 2
            obj_size = (box_3d[3:6] - box_3d[:3]) / 2

            delta_t = obj_center - obj_scan_coords
            delta_s = np.ones_like(delta_t) * obj_size
            w = 1 - (np.linalg.norm(delta_t / delta_s, axis=1, ord=2) / np.sqrt(3))

            reg_values[0, :, obj_scan_mask] = np.concatenate([np.expand_dims(w, 1), delta_t, delta_s], 1)

            bboxes.append(np.expand_dims(box_3d, 0))
            obj_idxs.append(obj_idx)

        sparse_reg_features.append(reg_values)
        sparse_box_features.append(bboxes)
        sparse_obj_features.append(obj_idxs)
        bscan_inst_mask.append(scan_inst_mask)
        bscan_obj.append(obj_feats)

        if (idx+1) % chunk_size == 0:
            bcoords = ME.utils.batched_coordinates(coords)
            bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
            sparse_chunk = ME.SparseTensor(bfeats.to(device),
                                              bcoords.to(device))  # Sparse tensor expects batched data
            dense_chunk = torch.unsqueeze(torch.cat(occ, dim=0), dim=1).to(device)
            sparse_features.append(sparse_chunk)
            dense_features.append(dense_chunk)
            coords = []
            feats = []
            occ = []

    return dense_features, sparse_features, (sparse_reg_features, sparse_box_features, sparse_obj_features, bscan_inst_mask, bscan_obj)