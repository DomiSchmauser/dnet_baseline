import sys
import os
import numpy as np
import json
import h5py
import torch

from torch.utils.data import Dataset

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from utils.data_utils import read_csv_mapping, load_hdf5, load_rgb, add_halfheight
from utils.pose_utils import backproject_rgb

class Front_dataset(Dataset):

    def __init__(self, base_dir, split='train'):

        self.split = split
        self.data_dir = os.path.join(base_dir, self.split)
        self.scenes = [f for f in os.listdir(os.path.abspath(self.data_dir))]
        mapping_file_path = os.path.join(base_dir, "3D_front_mapping.csv")
        _, self.csv_dict = read_csv_mapping(mapping_file_path)
        self.camera_intrinsics = np.array([[292.87803547399, 0, 0], [0, 292.87803547399, 0], [0, 0, 1]])

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
            record["pc_rgb"] = backproject_rgb(record["rgb"], record["depth_map"], self.camera_intrinsics)

            # Object level annotations
            for anno in imgs_anns['annotations']:

                if anno['image_id'] == v['id']:

                    cat_id = anno['category_id'] # 1 = chair, 2 = table, 3 = sofa, 4 = bed
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

                    obj = {
                        'category_id': cat_id,
                        'instance_id': instance_id,
                        'voxel_path': voxel_path,
                        'box_2d': box_2d,
                        'box_3d': box_3d,
                        'segmask': segmask,
                        'scale': scale,
                        'rot': rot_3d,
                        'loc': loc_3d
                    }
                    obj_anns.append(obj)

            record['obj_anns'] = obj_anns
            img_dict.append(record)


        return img_dict



