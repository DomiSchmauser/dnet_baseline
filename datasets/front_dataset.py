import sys
import os
import numpy as np
import json
import h5py
import torch
import MinkowskiEngine as ME
import open3d as o3d
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from utils.data_utils import read_csv_mapping, load_hdf5, load_rgb, add_halfheight, coords2occupancy, get_voxel, boxpt2voxel
from utils.pose_utils import backproject_rgb, cam2world, occ2noc, discretize_noc, occ2noc2
from utils.net_utils import vg_crop
from dvis import dvis

class Front_dataset(Dataset):

    def __init__(self, base_dir, split='train'):

        self.split = split
        self.data_dir = os.path.join(base_dir, self.split)
        self.scenes = [f for f in os.listdir(os.path.abspath(self.data_dir))]
        mapping_file_path = os.path.join(base_dir, "3D_front_mapping.csv")
        _, self.csv_dict = read_csv_mapping(mapping_file_path)
        self.camera_intrinsics = np.array([[292.87803547399, 0, 0], [0, 292.87803547399, 0], [0, 0, 1]])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Quantization and sparsify
        self.shift_pc = True
        self.padded_size = [192, 192, 96] # x y z
        self.quantization_size = 0.03
        self.debugging_mode = False

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        img_dict = []
        sequence = self.scenes[idx]

        json_file = os.path.join(self.data_dir, sequence, "coco_data/coco_annotations.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        # Image level annotations
        for idx, v in enumerate(imgs_anns['images']):

            record = {}
            obj_anns = []
            filename = os.path.join(self.data_dir, sequence, 'coco_data', v["file_name"])
            hdf5_path = os.path.join(self.data_dir, sequence, str(idx) + '.hdf5')
            rgb_path = os.path.join(self.data_dir, sequence, 'coco_data', 'rgb_' + str(idx).zfill(4) + '.png')

            record["file_name"] = filename
            record["height"] = v['height']
            record["width"] = v['width']
            record["depth_map"], record['campose'], cx, cy = load_hdf5(hdf5_path)
            self.camera_intrinsics[0, 2] = cx # check position
            self.camera_intrinsics[1, 2] = cy
            record["rgb"] = load_rgb(rgb_path)

            # Scene preparation
            cam_rgb_pc = backproject_rgb(record["rgb"], record["depth_map"], self.camera_intrinsics)
            record["pc_rgb"] = cam2world(cam_rgb_pc, record['campose'])

            if self.shift_pc:
                offset = record['pc_rgb'][:,:3].min(axis=0)
                record['pc_rgb'][:,:3] -= offset
                record['pc_offset'] = offset

            # Sparse Input
            record['sparse_coords'], record['sparse_feats'] = ME.utils.sparse_quantize(record['pc_rgb'][:,:3], features=record['pc_rgb'][:,3:], quantization_size=self.quantization_size)

            # Dense Input
            record['dense_grid'] = coords2occupancy(record['sparse_coords'], padded_size=self.padded_size, as_padded_whl=True)
            record['obj_scan_mask'] = torch.clone(record['dense_grid'])
            noc_shape = [3] + list(record['dense_grid'].shape)
            record['noc_scan_mask'] = torch.zeros(noc_shape)

            if self.debugging_mode:
                dvis(record['dense_grid'], c=0)

            # Object level annotations
            for anno in imgs_anns['annotations']:

                if anno['image_id'] == v['id']:

                    cat_id = anno['category_id'] # 1 = chair, 2 = table, 3 = sofa, 4 = bed
                    if cat_id == 2:
                        rot_sym = 'c2'
                    else:
                        rot_sym = 'None'

                    instance_id = int(anno['id']) + 2 # shift by 2 to avoid confusion 0 and 1 which represent occupancies
                    jid = anno['jid']
                    #print(jid)
                    #cat_name = self.csv_dict[cat_id]
                    voxel_path = os.path.join(CONF.PATH.FUTURE3D, jid, 'model.binvox')
                    box_2d = anno['bbox']
                    segmask = anno['segmentation']
                    # Pose
                    scale = np.array(anno['3Dscale'])
                    rot_3d = anno['3Drot']
                    box_3d = anno['3Dbbox']
                    loc_3d = add_halfheight(anno['3Dloc'], anno['3Dbbox'])

                    # Requires according shift of GT 3D location annotations
                    if self.shift_pc:
                        box_3d -= record['pc_offset']
                        box_3d[box_3d < 0] = 0 # if box is outside frame with some part clip box
                        loc_3d -= record['pc_offset']

                    box_3d = boxpt2voxel(box_3d, self.quantization_size)

                    # Use box 3d and occupancy values to index object scan mask
                    cropped_obj = vg_crop(record['dense_grid'].numpy(), box_3d)
                    #noc_obj = occ2noc2(cropped_obj, rot_3d)

                    if self.debugging_mode:
                        dvis(np.expand_dims(box_3d, axis=0), fmt='box', c=1)
                        #dvis(cropped_obj, c=1)

                    cropped_obj[cropped_obj != 0] = instance_id
                    record['obj_scan_mask'][int(box_3d[0]):int(box_3d[3]), int(box_3d[1]):int(box_3d[4]), int(box_3d[2]):int(box_3d[5])] = torch.from_numpy(cropped_obj)

                    # Load binvox
                    bin_vox = get_voxel(voxel_path, scale)

                    # Binvox to noc, then discretize and scale, finally place in the scene
                    noc = occ2noc(bin_vox, rot_3d) # as pc

                    noc = discretize_noc(noc, box_3d, quantization_size=self.quantization_size)


                    record['noc_scan_mask'][int(box_3d[0]):int(box_3d[3]), int(box_3d[1]):int(box_3d[4]),
                    int(box_3d[2]):int(box_3d[5])] = torch.from_numpy(cropped_obj)

                    '''
                    # Debugging
                    if self.debugging_mode:
                        nocs_pcd = o3d.geometry.PointCloud()
                        nocs_pcd.points = o3d.utility.Vector3dVector(noc)
                        nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
                        o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
                    '''

                    obj = {
                        'category_id': cat_id,
                        'instance_id': instance_id,
                        'voxel_path': voxel_path,
                        'box_2d': box_2d,
                        'box_3d': box_3d,
                        'segmask': segmask,
                        'scale': scale,
                        'rot': rot_3d,
                        'loc': loc_3d,
                        'rot_sym': rot_sym,
                        'noc': noc,
                        'voxel': bin_vox,
                    }
                    obj_anns.append(obj)

            record['obj_anns'] = obj_anns
            img_dict.append(record)

            '''
            # Debugging
            if self.debugging_mode and idx == 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(record['pc_rgb'][:,:3])
                nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
                boxes = []
                for ann in obj_anns:
                    vis_box = o3d.geometry.OrientedBoundingBox()
                    vis_box = vis_box.create_from_points(o3d.utility.Vector3dVector(ann['box_3d']))
                    boxes.append(vis_box)
                boxes.append(pcd)
                boxes.append(nocs_origin)
                o3d.visualization.draw_geometries(boxes)
            '''

        return img_dict



