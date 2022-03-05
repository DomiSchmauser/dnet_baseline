import numpy as np
import open3d as o3d
import mathutils
import MinkowskiEngine as ME
import torch

from torch.nn import functional as F
from sklearn.preprocessing import minmax_scale
from dvis import dvis

def get_noc2scan(rot_3d, loc_3d, scale, bin_vox, quantization_size=0.04):
    '''
    Calculates the noc2scan matrix
    Not in the discretized space
    '''

    euler = mathutils.Euler(rot_3d)
    rot = np.array(euler.to_matrix())

    # Cad2Scan #todo check scale quant only applied to translation or also as scale to translation
    scale_quant = 1/quantization_size
    cad2scan = np.identity(4)
    cad2scan[:3, :3] = np.diag(scale) @ rot
    cad2scan[:3, 3] = loc_3d * scale_quant

    # Noc2Cad
    noc2cad = np.identity(4)

    # Y axis starts at 0 in cad space and is not centered
    nonzero_inds = np.nonzero(bin_vox)[:-1]
    max_value = bin_vox.shape[0] - 1
    points = nonzero_inds / max_value
    shift_y = points.numpy()[:, 1].min()
    noc2cad[:3, 3] = np.array([-0.5, -shift_y, -0.5])

    cad2noc = np.linalg.inv(noc2cad)

    noc2scan = cad2scan @ noc2cad # cad2scan @ noc2cad = noc2scan?? -> lower with now

    return noc2scan, cad2noc

def occ2noc(cropped_obj, box_3d, euler_rot):
    '''
    Get occ coords of a cropped object, transform to noc space
    Height object is y coord
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

    '''
    nocs_pcd = o3d.geometry.PointCloud()
    nocs_pcd.points = o3d.utility.Vector3dVector(noc.T)
    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
    '''

    noc_obj = np.zeros((3, cropped_obj.shape[0], cropped_obj.shape[1], cropped_obj.shape[2]))
    noc_obj[:, crop_idxs[0], crop_idxs[1], crop_idxs[2]] = noc

    return noc_obj

def occ2world(voxel_grid, euler_rot, translation, bbox, quantization_size=0.03, max_extensions=None):
    '''
    euler rotation: CAD2World space in Blender coord space
    '''

    euler = mathutils.Euler(euler_rot)
    rot = np.array(euler.to_matrix())

    # Voxel space to CAD space
    nonzero_inds = np.nonzero(voxel_grid)[:-1]
    #max_value = voxel_grid.shape[0] - 1
    points = nonzero_inds / 31 - 0.5
    points = points.numpy()
    points[:, 1] -= points[:, 1].min() # CAD space y is shifted up to start at 0

    # CAD to world
    world_pc = rot @ points.transpose() + np.expand_dims(translation, axis=-1)
    world_pc = world_pc.transpose() # World space

    # Discretize
    coords = ME.utils.sparse_quantize(torch.from_numpy(world_pc).contiguous(memory_format=torch.contiguous_format),
                                      features=None,
                                      quantization_size=quantization_size)

    grid = np.zeros((192, 192, 96))

    #x = coords[:, 0]
    #y = coords[:, 1]
    #z = coords[:, 2]

    x_min, x_max = bbox[0], bbox[3]
    y_min, y_max = bbox[1], bbox[4]
    z_min, z_max = bbox[2], bbox[5]

    x_scaled = np.clip(np.rint(minmax_scale(coords[:, 0], feature_range=(x_min, x_max))).astype(np.int), 0, max_extensions[0])
    y_scaled = np.clip(np.rint(minmax_scale(coords[:, 1], feature_range=(y_min, y_max))).astype(np.int), 0, max_extensions[1])
    z_scaled = np.clip(np.rint(minmax_scale(coords[:, 2], feature_range=(z_min, z_max))).astype(np.int), 0, max_extensions[2])

    grid[x_scaled,y_scaled,z_scaled] = 1


    #out_size = tuple(bbox[3:].astype(np.int))

    # Interpolate
    #scaled_grid = F.interpolate(torch.from_numpy(np.expand_dims(grid, axis=(0,1))), size=out_size, mode='trilinear', align_corners=True)

    #dvis(np.expand_dims(bbox, axis=0), fmt='box', c=1)
    #dvis(grid, fmt='voxels')


    return (x_scaled, y_scaled, z_scaled)


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
        nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        depth_pc_obj.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw_geometries([depth_pc_obj, nocs_origin])


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

'''
def center(points):
    x_range = points[:,0].max() - points[:,0].min()
    y_range = points[:,1].max() - points[:,1].min()
    z_range = points[:,2].max() - points[:,2].min()
    ranges = np.array([x_range, y_range, z_range])
    return points - ranges/2
'''

'''
    nocs_pcd = o3d.geometry.PointCloud()
    nocs_pcd.points = o3d.utility.Vector3dVector(points)
    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
    '''

'''
 pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(world_pc)

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, 0.03)
    occ_list = vg.get_voxels()
    c = []
    for vx in occ_list:
        coordinates = vx.grid_index
        c.append(np.expand_dims(coordinates, axis=0))

    vg_idxs = np.concatenate(c, axis=0)
    x_m, y_m, z_m = vg_idxs[:,0].max(), vg_idxs[:,1].max(), vg_idxs[:,2].max()
    grid = torch.zeros((x_m+1, y_m+1, z_m+1))

    grid[vg_idxs[:,0], vg_idxs[:,1], vg_idxs[:,2]] = 1
    dvis(grid, fmt='voxels')
'''

'''

 # Shift to zero
 box_min = bbox[:3].copy()
 bbox[:3] -= box_min
 bbox[3:] -= box_min
 bbox = bbox.astype(np.int)

 grid = np.zeros((bbox[3], bbox[4], bbox[5]))


 x_scaled = np.rint(minmax_scale(coords[:, 0], feature_range=(0, bbox[3]-1))).astype(np.int)
 y_scaled = np.rint(minmax_scale(coords[:, 1], feature_range=(0, bbox[4]-1))).astype(np.int)
 z_scaled = np.rint(minmax_scale(coords[:, 2], feature_range=(0, bbox[5]-1))).astype(np.int)
 grid[x_scaled, y_scaled, z_scaled] = 1

 '''
'''
# Shift to 0
x -= x.min()
y -= y.min()
z -= z.min()

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()

grid = np.zeros((x_max+1, y_max+1, z_max+1))
'''
