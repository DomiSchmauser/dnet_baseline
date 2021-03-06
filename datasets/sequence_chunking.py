import torch
import numpy as np
import MinkowskiEngine as ME
from torch.nn import functional as F
import open3d as o3d

def batch_collate_cpu(batch):
    '''
    returns:
    dense_features: list of n chunks with batched features shape: BS(num imgs) x dims
    sparse_features: list of n chunks with batched sparse tensors
    sparse_reg_features: list of per image features, len=25
    '''

    occ = []
    coords = []
    feats = []
    sparse_reg_features = []
    sparse_obj_features = []
    sparse_box_features = []
    bscan_inst_mask = []
    bscan_nocs_mask = []
    bscan_obj = []
    bscan_info = []

    for idx, img in enumerate(batch):

        # Empty object proposals in image due to outside boxes
        if img is None:
            continue

        # Scan level Level
        img_grid = torch.unsqueeze(img['dense_grid'], dim=0)
        occ.append(img_grid)
        coords.append(img['sparse_coords'])
        feats.append(img['sparse_feats'])
        bscan_info.append(img['rgb_path'])

        # Object level
        reg_values = np.zeros([1, 7, img['dense_grid'].shape[0], img['dense_grid'].shape[1], img['dense_grid'].shape[2]])
        scan_inst_mask = img['obj_scan_mask']
        nocs_mask = img['noc_scan_mask']
        bboxes = []
        obj_idxs = []
        obj_feats = {}

        for obj in img['obj_anns']:
            box_3d = obj['box_3d']
            obj_idx = obj['instance_id']

            # Dense reg values
            obj_scan_mask = scan_inst_mask == int(obj_idx)
            obj_scan_mask = obj_scan_mask.numpy()
            obj_scan_coords = np.argwhere(obj_scan_mask)

            obj_center = (box_3d[3:6] + box_3d[:3]) / 2
            obj_size = (box_3d[3:6] - box_3d[:3]) / 2

            delta_t = obj_center - obj_scan_coords
            delta_s = np.ones_like(delta_t) * obj_size
            w = 1 - (np.linalg.norm(delta_t / delta_s, axis=1, ord=2) / np.sqrt(3))  # W represents distance to center for objectness supervision

            reg_values[0, :, obj_scan_mask] = np.concatenate([np.expand_dims(w, 1), delta_t, delta_s], 1)

            # Fill in object features
            obj_feats[str(obj_idx)] = {}
            obj_feats[str(obj_idx)]['class_name'] = obj['class_name']
            obj_feats[str(obj_idx)]['category_id'] = obj['category_id']
            obj_feats[str(obj_idx)]['num_occ'] = obj_scan_mask.sum()  # counts True elements in scan
            obj_feats[str(obj_idx)]['noc2scan'] = torch.from_numpy(obj['noc2scan'])
            obj_feats[str(obj_idx)]['rot_sym'] = obj['rot_sym']
            obj_feats[str(obj_idx)]['aligned2scan'] = torch.from_numpy(obj['cad2scan']).to(torch.float32)
            obj_feats[str(obj_idx)]['aligned2noc'] = torch.from_numpy(obj['cad2noc']).to(torch.float32)
            obj_feats[str(obj_idx)]['box_3d'] = box_3d

            bboxes.append(np.expand_dims(box_3d, 0))
            obj_idxs.append(obj_idx)

        sparse_reg_features.append(reg_values)
        sparse_box_features.append(torch.from_numpy(np.concatenate(bboxes, axis=0)))
        sparse_obj_features.append(obj_idxs)
        bscan_inst_mask.append(torch.unsqueeze(scan_inst_mask, dim=0))  # Move to cuda
        bscan_nocs_mask.append(nocs_mask)  # Move to cuda
        bscan_obj.append(obj_feats)

    # Batch sparse and dense features
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    dense_features = torch.unsqueeze(torch.cat(occ, dim=0), dim=1).to(torch.float)

    # Batch sparse regression features
    sparse_reg_features = torch.from_numpy(np.concatenate(sparse_reg_features, axis=0))
    s_feats = []
    for s_idx, s_coord in enumerate(coords):
        s_reg_feats = sparse_reg_features[s_idx, :, s_coord[:, 0].long(), s_coord[:, 1].long(), s_coord[:, 2].long()]
        s_feats.append(torch.transpose(s_reg_feats, 0, 1))

    bs_reg_feats = torch.cat(s_feats, dim=0)
    sparse_reg_tensor = (bs_reg_feats, coords)
    sparse_features = (bfeats, coords)

    return dense_features, sparse_features, \
           (sparse_reg_tensor, sparse_box_features, sparse_obj_features,
            bscan_inst_mask, bscan_nocs_mask, bscan_obj,bscan_info)


def batch_collate(batch):
    '''
    returns:
    dense_features: list of n chunks with batched features shape: BS(num imgs) x dims
    sparse_features: list of n chunks with batched sparse tensors
    sparse_reg_features: list of per image features, len=25
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    occ = []
    coords = []
    feats = []
    sparse_reg_features = []
    sparse_obj_features = []
    sparse_box_features = []
    bscan_inst_mask = []
    bscan_nocs_mask = []
    bscan_obj = []
    bscan_info = []

    for idx, img in enumerate(batch):

        # Empty object proposals in image due to outside boxes
        if img is None:
            continue

        # Scan level Level
        img_grid = torch.unsqueeze(img['dense_grid'], dim=0)
        occ.append(img_grid)
        coords.append(img['sparse_coords'])
        feats.append(img['sparse_feats'])
        bscan_info.append(img['rgb_path'])

        # Object level
        reg_values = np.zeros([1, 7, img['dense_grid'].shape[0], img['dense_grid'].shape[1], img['dense_grid'].shape[2]])
        scan_inst_mask = img['obj_scan_mask']
        nocs_mask = img['noc_scan_mask']
        bboxes = []
        obj_idxs = []
        obj_feats = {}

        for obj in img['obj_anns']:

            box_3d = obj['box_3d']
            obj_idx = obj['instance_id']

            # Dense reg values
            obj_scan_mask = scan_inst_mask == int(obj_idx)
            '''
            # dilate for sparse overlap
            # TODO PRECALC FOR SPEEDUP
            obj_scan_mask_torch = obj_scan_mask.cuda()
            obj_scan_mask_torch2 = F.conv3d(
                obj_scan_mask_torch.float().unsqueeze(0).unsqueeze(0), weight=torch.ones(1, 1, 9, 9, 9).cuda(),
                padding=4
            )[0][0]
            obj_scan_mask = (obj_scan_mask_torch2 > 0).cpu().numpy()
            '''
            obj_scan_mask = obj_scan_mask.numpy()

            obj_scan_coords = np.argwhere(obj_scan_mask)

            obj_center = (box_3d[3:6] + box_3d[:3]) / 2
            obj_size = (box_3d[3:6] - box_3d[:3]) / 2

            delta_t = obj_center - obj_scan_coords
            delta_s = np.ones_like(delta_t) * obj_size
            w = 1 - (np.linalg.norm(delta_t / delta_s, axis=1, ord=2) / np.sqrt(3)) # W represents distance to center for objectness supervision

            reg_values[0, :, obj_scan_mask] = np.concatenate([np.expand_dims(w, 1), delta_t, delta_s], 1)

            # Fill in object features
            obj_feats[str(obj_idx)] = {}
            obj_feats[str(obj_idx)]['class_name'] = obj['class_name']
            obj_feats[str(obj_idx)]['category_id'] = obj['category_id']
            obj_feats[str(obj_idx)]['num_occ'] = obj_scan_mask.sum()  # counts True elements in scan
            obj_feats[str(obj_idx)]['noc2scan'] = torch.from_numpy(obj['noc2scan']).to(device)
            obj_feats[str(obj_idx)]['rot_sym'] = obj['rot_sym']
            obj_feats[str(obj_idx)]['aligned2scan'] = torch.from_numpy(obj['cad2scan']).to(torch.float32).to(device)
            obj_feats[str(obj_idx)]['aligned2noc'] = torch.from_numpy(obj['cad2noc']).to(torch.float32)
            obj_feats[str(obj_idx)]['box_3d'] = box_3d

            bboxes.append(np.expand_dims(box_3d, 0))
            obj_idxs.append(obj_idx)

        sparse_reg_features.append(reg_values)
        sparse_box_features.append(torch.from_numpy(np.concatenate(bboxes, axis=0)).to(device))
        sparse_obj_features.append(obj_idxs)
        bscan_inst_mask.append(torch.unsqueeze(scan_inst_mask, dim=0).to(device)) # Move to cuda
        bscan_nocs_mask.append(nocs_mask.to(device)) # Move to cuda
        bscan_obj.append(obj_feats)

    # Batch sparse and dense features
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    sparse_features = ME.SparseTensor(bfeats.to(device),
                                      bcoords.to(device))  # Sparse tensor expects batched data
    dense_features = torch.unsqueeze(torch.cat(occ, dim=0), dim=1).to(device).to(torch.float)

    # Batch sparse regression features
    sparse_reg_features = torch.from_numpy(np.concatenate(sparse_reg_features, axis=0))
    s_coords, _ = sparse_features.decomposed_coordinates_and_features  # for decomposition of batched data
    s_feats = []
    s_cpu_coords = []
    for s_idx, s_coord in enumerate(s_coords):

        s_reg_feats = sparse_reg_features[s_idx, :, s_coord[:, 0].long(), s_coord[:, 1].long(), s_coord[:, 2].long()]
        s_feats.append(torch.transpose(s_reg_feats, 0, 1))
        s_cpu_coords.append(s_coord.detach().cpu())

    bs_reg_feats = torch.cat(s_feats, dim=0)
    bs_reg_coords = ME.utils.batched_coordinates(s_cpu_coords)
    sparse_reg_tensor = ME.SparseTensor(bs_reg_feats.to(device),
                                        bs_reg_coords.to(device))


    return dense_features, sparse_features, (sparse_reg_tensor, sparse_box_features, sparse_obj_features, bscan_inst_mask, bscan_nocs_mask, bscan_obj, bscan_info)

def batch_collate_infer(batch):
    '''
    returns:
    dense_features: list of n chunks with batched features shape: BS(num imgs) x dims
    sparse_features: list of n chunks with batched sparse tensors
    sparse_reg_features: list of per image features, len=25
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    occ = []
    coords = []
    feats = []
    sparse_reg_features = []
    sparse_obj_features = []
    sparse_box_features = []
    bscan_inst_mask = []
    bscan_nocs_mask = []
    bscan_obj = []
    bscan_info = []
    bcam_info = []
    bshift_info = []
    b_pc = []


    for idx, img in enumerate(batch):

        # Empty object proposals in image due to outside boxes
        if img is None:
            continue

        # Scan level Level
        img_grid = torch.unsqueeze(img['dense_grid'], dim=0)
        occ.append(img_grid)
        coords.append(img['sparse_coords'])
        feats.append(img['sparse_feats'])
        bscan_info.append(img['rgb_path'])
        bcam_info.append(img['campose'])
        bshift_info.append(img['pc_offset'])
        b_pc.append(img['pc_rgb'])

        # Object level
        reg_values = np.zeros([1, 7, img['dense_grid'].shape[0], img['dense_grid'].shape[1], img['dense_grid'].shape[2]])
        scan_inst_mask = img['obj_scan_mask']
        nocs_mask = img['noc_scan_mask']
        bboxes = []
        obj_idxs = []
        obj_feats = {}

        for obj in img['obj_anns']:

            box_3d = obj['box_3d']
            obj_idx = obj['instance_id']

            # Dense reg values
            obj_scan_mask = scan_inst_mask == int(obj_idx)
            '''
            # dilate for sparse overlap
            # TODO PRECALC FOR SPEEDUP
            obj_scan_mask_torch = obj_scan_mask.cuda()
            obj_scan_mask_torch2 = F.conv3d(
                obj_scan_mask_torch.float().unsqueeze(0).unsqueeze(0), weight=torch.ones(1, 1, 9, 9, 9).cuda(),
                padding=4
            )[0][0]
            obj_scan_mask = (obj_scan_mask_torch2 > 0).cpu().numpy()
            '''
            obj_scan_mask = obj_scan_mask.numpy()

            obj_scan_coords = np.argwhere(obj_scan_mask)

            obj_center = (box_3d[3:6] + box_3d[:3]) / 2
            obj_size = (box_3d[3:6] - box_3d[:3]) / 2

            delta_t = obj_center - obj_scan_coords
            delta_s = np.ones_like(delta_t) * obj_size
            w = 1 - (np.linalg.norm(delta_t / delta_s, axis=1, ord=2) / np.sqrt(3)) # W represents distance to center for objectness supervision

            reg_values[0, :, obj_scan_mask] = np.concatenate([np.expand_dims(w, 1), delta_t, delta_s], 1)

            # Fill in object features
            obj_feats[str(obj_idx)] = {}
            obj_feats[str(obj_idx)]['class_name'] = obj['class_name']
            obj_feats[str(obj_idx)]['category_id'] = obj['category_id']
            obj_feats[str(obj_idx)]['num_occ'] = obj_scan_mask.sum()  # counts True elements in scan
            obj_feats[str(obj_idx)]['rot_sym'] = obj['rot_sym']
            obj_feats[str(obj_idx)]['noc2scan'] = torch.from_numpy(obj['noc2scan']).to(device)
            obj_feats[str(obj_idx)]['aligned2scan'] = torch.from_numpy(obj['cad2scan']).to(torch.float32).to(device)
            obj_feats[str(obj_idx)]['cad2world'] = torch.from_numpy(obj['cad2world']).to(torch.float32).to(device)
            obj_feats[str(obj_idx)]['aligned2noc'] = torch.from_numpy(obj['cad2noc']).to(torch.float32)
            obj_feats[str(obj_idx)]['box_3d'] = box_3d

            bboxes.append(np.expand_dims(box_3d, 0))
            obj_idxs.append(obj_idx)

        sparse_reg_features.append(reg_values)
        sparse_box_features.append(torch.from_numpy(np.concatenate(bboxes, axis=0)).to(device))
        sparse_obj_features.append(obj_idxs)
        bscan_inst_mask.append(torch.unsqueeze(scan_inst_mask, dim=0).to(device)) # Move to cuda
        bscan_nocs_mask.append(nocs_mask.to(device)) # Move to cuda
        bscan_obj.append(obj_feats)

    if not coords:
        return None, None, (None, sparse_box_features, sparse_obj_features, bscan_inst_mask, bscan_nocs_mask, bscan_obj, bscan_info, bcam_info, bshift_info)
    # Batch sparse and dense features
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    sparse_features = ME.SparseTensor(bfeats.to(device),
                                      bcoords.to(device))  # Sparse tensor expects batched data
    dense_features = torch.unsqueeze(torch.cat(occ, dim=0), dim=1).to(device).to(torch.float)

    # Batch sparse regression features
    sparse_reg_features = torch.from_numpy(np.concatenate(sparse_reg_features, axis=0))
    s_coords, _ = sparse_features.decomposed_coordinates_and_features  # for decomposition of batched data
    s_feats = []
    s_cpu_coords = []
    for s_idx, s_coord in enumerate(s_coords):

        s_reg_feats = sparse_reg_features[s_idx, :, s_coord[:, 0].long(), s_coord[:, 1].long(), s_coord[:, 2].long()]
        s_feats.append(torch.transpose(s_reg_feats, 0, 1))
        s_cpu_coords.append(s_coord.detach().cpu())

    bs_reg_feats = torch.cat(s_feats, dim=0)
    bs_reg_coords = ME.utils.batched_coordinates(s_cpu_coords)
    sparse_reg_tensor = ME.SparseTensor(bs_reg_feats.to(device),
                                        bs_reg_coords.to(device))


    return dense_features, sparse_features, (sparse_reg_tensor, sparse_box_features, sparse_obj_features, bscan_inst_mask, bscan_nocs_mask, bscan_obj, bscan_info, bcam_info, bshift_info, b_pc)



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
    bscan_nocs_mask = []
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
        nocs_mask = img['noc_scan_mask']
        bboxes = []
        obj_idxs = []
        obj_feats = {}

        for obj in img['obj_anns']:

            box_3d = obj['box_3d']
            obj_idx = obj['instance_id']

            obj_scan_mask = scan_inst_mask == int(obj_idx)
            obj_scan_mask = obj_scan_mask.numpy()

            # Fill in object features
            obj_feats[str(obj_idx)] = {}
            obj_feats[str(obj_idx)]['num_occ'] = obj_scan_mask.sum()# counts True elements in scan
            obj_feats[str(obj_idx)]['noc2scan'] = torch.from_numpy(obj['noc2scan']).to(device)
            obj_feats[str(obj_idx)]['rot_sym'] = obj['rot_sym']
            obj_feats[str(obj_idx)]['aligned2scan'] = torch.eye(4).to(device)
            obj_feats[str(obj_idx)]['aligned2scan'][:3,3] = torch.from_numpy(obj['loc'])
            obj_feats[str(obj_idx)]['aligned2scan'][:3,:3] = torch.from_numpy(obj['rot'])
            obj_feats[str(obj_idx)]['aligned2noc'] = torch.from_numpy(obj['cad2noc']).to(torch.float32)


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
        sparse_box_features.append(torch.from_numpy(np.concatenate(bboxes, axis=0)).to(device))
        sparse_obj_features.append(obj_idxs)
        bscan_inst_mask.append(torch.unsqueeze(scan_inst_mask, dim=0).to(device)) # Move to cuda
        bscan_nocs_mask.append(nocs_mask.to(device)) # Move to cuda
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

    return dense_features, sparse_features, (sparse_reg_features, sparse_box_features, sparse_obj_features, bscan_inst_mask, bscan_nocs_mask, bscan_obj)