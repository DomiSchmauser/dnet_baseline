import csv, os
import cv2
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.voxel_utils import binvox_rw


def read_csv_mapping(path):
    """
     Loads an idset mapping from a csv file, assuming the rows are sorted by their ids.
    :param path: Path to csv file
    """

    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        new_id_label_map = []
        new_label_id_map = {}

        for row in reader:
            new_id_label_map.append(row["name"])
            new_label_id_map[int(row["id"])] = row["name"]

        return new_id_label_map, new_label_id_map

def load_hdf5(hdf5_path):
    '''
    Loads campose and depth map from an hdf5 file
    returns additional camera intrinsics cx, cy
    '''

    with h5py.File(hdf5_path, 'r') as data:
        for key in data.keys():
            if key == 'depth':
                depth = np.array(data[key])
            elif key == 'campose':
                campose = np.array(data[key])

    img_width = depth.shape[1]
    img_height = depth.shape[0]

    cx = (img_width / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    cy = (img_height / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5

    return depth, campose, cx, cy

def load_rgb(rgb_path):
    '''
    Loads a rgb image from a png file
    '''

    bgr_img = cv2.imread(rgb_path)
    rgb_img = bgr_img[:,:,::-1]
    rgb_img = np.array(rgb_img, dtype=np.float32)

    return rgb_img

def get_voxel(voxel_path, scale):
    '''
    Load voxel grid and rescale with according scale parameters if scale is any other than [1, 1, 1]
    voxel_path: path to 3D-FUTURE model
    scale: array with scale parameter in xyz
    '''
    if not os.path.exists(voxel_path):
        raise ValueError('Voxelized model does not exist for this path!', voxel_path)

    with open(voxel_path, 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f).data

    unscaled_voxel = voxel.astype(int)

    if np.all(scale == 1): # No scaling required
        rescaled_voxel = unscaled_voxel
    else:
        rescaled_voxel = rescale_voxel(unscaled_voxel, scale)

    return torch.from_numpy(rescaled_voxel)

def rescale_voxel(unscaled_voxel, scale, debug_mode=False):
    '''
    Rescale 3D voxel grid by a given scale array
    '''

    centering = unscaled_voxel.shape[0] / 2
    max_value = unscaled_voxel.shape[0] - 1
    non_zeros = np.nonzero(unscaled_voxel)
    scale_mat = np.diag(scale)
    xyz = (np.stack(non_zeros, axis=0).T - centering) @ (scale_mat / scale.max())
    xyz = np.rint(xyz) + centering
    xyz[xyz>max_value] = max_value # all values rounded up to 32 are set to max -> 31
    x = xyz[:,0].astype(np.int32)
    y = xyz[:,1].astype(np.int32)
    z = xyz[:,2].astype(np.int32)
    rescale_ = np.zeros(unscaled_voxel.shape)
    rescale_[x, y, z] = 1

    if debug_mode:
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(rescale_, edgecolor='k')
        #ax.voxels(unscaled_voxel, edgecolor='k')
        plt.show()
        ax = plt.figure().add_subplot(projection='3d')
        #ax.voxels(rescale_, edgecolor='k')
        ax.voxels(unscaled_voxel, edgecolor='k')
        plt.show()

    return rescale_

def add_halfheight(location, box):
    '''
    Object location z-center is at bottom, calculate half height of the object
    and add to shift z-center to correct location
    '''
    z_coords = []
    for pt in box:
        z = pt[-1]
        z_coords.append(z)
    z_coords = np.array(z_coords)
    half_height = np.abs(z_coords.max() - z_coords.min()) / 2
    location[-1] = half_height  # Center location is at bottom object

    return location