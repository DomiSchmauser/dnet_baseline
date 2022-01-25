import numpy as np
import open3d as o3d


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