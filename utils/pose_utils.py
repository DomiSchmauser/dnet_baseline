import numpy as np
import open3d as o3d
import mathutils
import MinkowskiEngine as ME
import torch

from sklearn.preprocessing import minmax_scale
from dvis import dvis

def get_noc2scan(rot_3d, loc_3d, scale, bin_vox):
    '''
    Calculates the noc2scan matrix
    Not in the discretized space
    '''

    euler = mathutils.Euler(rot_3d)
    rot = np.array(euler.to_matrix())

    # Cad2Scan
    cad2scan = np.identity(4)
    cad2scan[:3, :3] = rot
    cad2scan[:3, 3] = loc_3d

    # Noc2Cad
    noc2cad = np.identity(4)
    noc2cad[:3, :3] = np.diag(scale) @ noc2cad[:3, :3]
    cad2noc = np.linalg.inv(noc2cad)

    # Y axis starts at 0 in cad space and is not centered
    nonzero_inds = np.nonzero(bin_vox)[:-1]
    points = nonzero_inds / 31
    shift_y = points.numpy()[:, 1].min()
    noc2cad[:3, 3] = np.array([-0.5, -shift_y, -0.5])

    noc2scan = noc2cad @ cad2scan

    return noc2scan, cad2noc

def occ2noc(cropped_obj, box_3d, euler_rot):
    '''
    Get occ coords of a cropped object, transform to noc space
    '''
    euler = mathutils.Euler(euler_rot)
    rot = np.array(euler.to_matrix())

    xyz_min = box_3d[:3]
    crop_idxs = np.nonzero(cropped_obj)
    occ_idxs = np.vstack((crop_idxs[0], crop_idxs[1], crop_idxs[2])).T + xyz_min.astype(np.int)

    # World to CAD rotation
    world2cad = (np.linalg.inv(rot) @ occ_idxs.T).T

    # To NOC space transform
    x_min, x_max = world2cad[:,0].min(), world2cad[:,0].max()
    y_min, y_max = world2cad[:,1].min(), world2cad[:,1].max()
    z_min, z_max = world2cad[:,2].min(), world2cad[:,2].max()
    max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max()

    x_noc = (world2cad[:,0] - ((x_max + x_min) / 2)) / max_range + 0.5
    y_noc = (world2cad[:,1] - ((y_max + y_min) / 2)) / max_range + 0.5
    z_noc = (world2cad[:,2] - ((z_max + z_min) / 2)) / max_range + 0.5

    noc = np.vstack((x_noc, y_noc, z_noc))

    noc_obj = np.zeros((3, cropped_obj.shape[0], cropped_obj.shape[1], cropped_obj.shape[2]))
    noc_obj[:, crop_idxs[0], crop_idxs[1], crop_idxs[2]] = noc

    return noc_obj

def occ2world(voxel_grid, euler_rot, translation, bbox, quantization_size=0.03):
    '''
    Calculate nocs map from occupancy grid and 3D rot
    euler rotation: CAD2World space in Blender coord space

    def center(points):
        x_range = points[:,0].max() - points[:,0].min()
        y_range = points[:,1].max() - points[:,1].min()
        z_range = points[:,2].max() - points[:,2].min()
        ranges = np.array([x_range, y_range, z_range])
        return points - ranges/2
    '''

    euler = mathutils.Euler(euler_rot)
    rot = np.array(euler.to_matrix())

    nonzero_inds = np.nonzero(voxel_grid)[:-1]

    max_idx = torch.max(nonzero_inds)

    points = nonzero_inds / 31 - 0.5
    points = points.numpy()
    points[:, 1] -= points[:, 1].min() # CAD space y is shifted up to start at 0

    '''
    nocs_pcd = o3d.geometry.PointCloud()
    nocs_pcd.points = o3d.utility.Vector3dVector(points)
    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
    '''

    # CAD to world
    world_pc = rot @ points.transpose() + np.expand_dims(translation, axis=-1)
    world_pc = world_pc.transpose() # world space

    # Discretize
    coords = ME.utils.sparse_quantize(torch.from_numpy(world_pc).contiguous(memory_format=torch.contiguous_format) , features=None,
                             quantization_size=quantization_size)

    x_min, x_max = bbox[0], bbox[3]
    y_min, y_max = bbox[1], bbox[4]
    z_min, z_max = bbox[2], bbox[5]

    x_scaled = minmax_scale(coords[:, 0], feature_range=(x_min, x_max))
    y_scaled = minmax_scale(coords[:, 1], feature_range=(y_min, y_max))
    z_scaled = minmax_scale(coords[:, 2], feature_range=(z_min, z_max))

    noc_discretized = np.rint(np.array([x_scaled, y_scaled, z_scaled]).T)


    #dvis(np.expand_dims(bbox, axis=0), fmt='box', c=1)
    #dvis(noc_discretized)

    return world_pc


def backproject_rgb(rgb, depth, intrinsics, debug_mode=False):
    '''
    Backproject depth map to camera space, with additional rgb values
    Returns: Depth PC with according RGB values in camspace, used idxs in pixel space
    '''

    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = (depth > 0)

    idxs = np.where(non_zero_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = -pts[:, 2]

    rgb_vals = rgb[idxs[0], idxs[1]]

    rgb_pts = np.concatenate((pts, rgb_vals), axis=-1)

    if debug_mode:
        depth_pc_obj = o3d.geometry.PointCloud()
        depth_pc_obj.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw_geometries([depth_pc_obj])


    return rgb_pts

def cam2world(rgb_pts, campose):
    '''
    transform camera space pc to world space pc
    '''
    trans = campose[:3, 3:]
    rot = campose[:3, :3]

    cam_pts = rgb_pts[:,:3]
    world_pc = np.dot(rot, cam_pts.transpose()) + trans
    world_pc = world_pc.transpose()

    rgb_world = np.concatenate((world_pc, rgb_pts[:,3:]), axis=-1)

    return rgb_world


