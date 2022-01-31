import sys
import os
import numpy as np
import json
import h5py
import torch
import MinkowskiEngine as ME
import open3d as o3d

from torch.utils.data import Dataset

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from utils.data_utils import read_csv_mapping, load_hdf5, load_rgb, add_halfheight, coords2occupancy, get_voxel
from utils.pose_utils import backproject_rgb, cam2world, occ2noc

class Front_dataset(Dataset):

    def __init__(self, base_dir, split='train'):

        self.split = split
        self.data_dir = os.path.join(base_dir, self.split)
        self.scenes = [f for f in os.listdir(os.path.abspath(self.data_dir))]
        mapping_file_path = os.path.join(base_dir, "3D_front_mapping.csv")
        _, self.csv_dict = read_csv_mapping(mapping_file_path)
        self.camera_intrinsics = np.array([[292.87803547399, 0, 0], [0, 292.87803547399, 0], [0, 0, 1]])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.shift_pc = True
        self.padded_size = [192, 96, 192]

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
            record['sparse_coords'], record['sparse_feats'] = ME.utils.sparse_quantize(record['pc_rgb'][:,:3], features=record['pc_rgb'][:,3:], quantization_size=0.03)

            # Dense Input
            record['dense_grid'] = coords2occupancy(record['sparse_coords'], padded_size=self.padded_size, as_padded_whl=True)

            # Object level annotations
            for anno in imgs_anns['annotations']:

                if anno['image_id'] == v['id']:

                    cat_id = anno['category_id'] # 1 = chair, 2 = table, 3 = sofa, 4 = bed
                    if cat_id == 2:
                        rot_sym = 'c2'
                    else:
                        rot_sym = 'None'
                    instance_id = anno['id']
                    jid = anno['jid']
                    cat_name = self.csv_dict[cat_id]
                    voxel_path = os.path.join(CONF.PATH.FUTURE3D, jid, 'model.binvox')
                    box_2d = anno['bbox']
                    segmask = anno['segmentation']
                    box_3d = anno['3Dbbox']
                    # Pose
                    scale = np.array(anno['3Dscale'])
                    rot_3d = anno['3Drot']
                    loc_3d = add_halfheight(anno['3Dloc'], anno['3Dbbox'])

                    # Load binvox
                    bin_vox = get_voxel(voxel_path, scale)

                    # Binvox to noc
                    noc = occ2noc(bin_vox, rot_3d)

                    # Debugging
                    '''
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

        return img_dict



