import csv, os
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import MinkowskiEngine as ME

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

def pcd2occupancy(pcd, max_ext=96):
    '''
    pcd: in range 0 to max
    Creates a dense occupancy grid from a Nx3 pointcloud, preserves relative size between x,y,z
    max_extension is the maximal extension of the final grid
    '''

    occupancy_grid = np.zeros((max_ext, max_ext, max_ext))
    max_val= pcd.max()
    scaling = max_ext / max_val
    grids_idxs = np.rint(pcd * scaling).astype(int)
    grids_idxs[grids_idxs > max_ext-1] = max_ext-1 # all values rounded up to max_ext are set to max_ext-1
    occupancy_grid[grids_idxs] = 1

    return occupancy_grid

def coords2occupancy(coords, as_padded_whl=True, padded_size=[192, 96, 192], debug_mode=False):
    '''
    coords: sparse coords of voxel occupancy values
    Creates a dense occupancy grid from sparse coords
    returns: occupancy grid with shape x,y,z -> for W x H x L switch dim z and y
    '''

    max_extensions = torch.max(coords, dim=0).values + 1
    occupancy_grid = torch.zeros(max_extensions.tolist())
    occ_idxs = coords.type(torch.LongTensor)
    x = occ_idxs[:,0]
    y = occ_idxs[:,1]
    z = occ_idxs[:,2]
    occupancy_grid[x, y, z] = 1

    if debug_mode:
        np_grid = occupancy_grid.detach().cpu().numpy()
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(np_grid, edgecolor='k')
        plt.show()

    if as_padded_whl:
        occupancy_grid = xyz2whl(occupancy_grid, padded_size=padded_size)

    return occupancy_grid

def xyz2whl(occupancy_grid, padded_size=None):
    '''
    transfroms a xyz occupancy grid of changing size to fixed size grid of padded size in the format whl
    '''

    whl_grid = occupancy_grid.permute(0, 2, 1).contiguous()

    if padded_size is not None:

        pad_diff = np.array(padded_size) - whl_grid.shape
        paddings = np.max([pad_diff, np.zeros(3)], 0).astype(int)
        p2d = [(0, i) for i in paddings]
        whl_grid = np.pad(whl_grid.numpy(), p2d, "constant", constant_values=3 * [(0, 0)])

    return torch.from_numpy(whl_grid)



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