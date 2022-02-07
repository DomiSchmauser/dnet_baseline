import numpy as np
import open3d as o3d
import mathutils
from sklearn.preprocessing import minmax_scale

def occ2noc2(voxel_grid, euler_rot):
    '''
    Calculate nocs map from occupancy grid and 3D rot
    Carefull CAD space is moved up in z axis when rotation for nocs space might result in an issue
    '''

    euler = mathutils.Euler(euler_rot)
    rot = np.array(euler.to_matrix())

    shape = voxel_grid.shape
    max_dim = np.array(shape).max() - 1 # idx starts at 0

    nonzero_inds = np.nonzero(voxel_grid)

    points = nonzero_inds / max_dim  # norm 0 - 1
    points = points.numpy()



    world_pc = rot @ points.transpose()
    world_pc = world_pc.transpose()

    return world_pc

def occ2noc(voxel_grid, euler_rot):
    '''
    Calculate nocs map from occupancy grid and 3D rot
    euler rotation: CAD2World space in Blender coord space
    '''

    def center(points):
        x_range = points[:,0].max() - points[:,0].min()
        y_range = points[:,1].max() - points[:,1].min()
        z_range = points[:,2].max() - points[:,2].min()
        ranges = np.array([x_range, y_range, z_range])
        return points - ranges/2

    euler = mathutils.Euler(euler_rot)
    rot = np.array(euler.to_matrix())

    nonzero_inds = np.nonzero(voxel_grid)[:-1]

    points = nonzero_inds / 31 - 0.5
    points = points.numpy()
    points[:, 1] -= points[:, 1].min() # CAD space y is shifted up to start at 0

    '''
    nocs_pcd = o3d.geometry.PointCloud()
    nocs_pcd.points = o3d.utility.Vector3dVector(points)
    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
    '''

    noc_pc = rot @ points.transpose()
    noc_pc = noc_pc.transpose() + 0.5 # NOC space

    return noc_pc

def discretize_noc(noc_pts, bbox, quantization_size=0.03):
    '''
    Discretize nocs pointcloud and squash between object box coordinates
    Maybe issue since box is axis aligned
    '''

    scaling = 1 / quantization_size
    x_min, x_max = bbox[0], bbox[3]
    y_min, y_max = bbox[1], bbox[4]
    z_min, z_max = bbox[2], bbox[5]

    noc_discretized = np.rint(noc_pts * scaling)
    x_scaled = minmax_scale(noc_discretized[:,0], feature_range=(x_min, x_max))
    y_scaled = minmax_scale(noc_discretized[:,1], feature_range=(y_min, y_max))
    z_scaled = minmax_scale(noc_discretized[:,2], feature_range=(z_min, z_max))

    noc_discretized = np.array([x_scaled, y_scaled, z_scaled]).T
    test = noc_discretized

    return noc_discretized



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


