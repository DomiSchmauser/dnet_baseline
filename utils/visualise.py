import sys
import time
import numpy as np
import torch
from dvis import dvis
import open3d as o3d

import trimesh
import mcubes
from sklearn.preprocessing import minmax_scale
from utils.net_utils import vg_crop


def visualise_pred_sequence(pred_trajectories, seq_name=None, seq_len=125, with_box=False, as_mesh=True, pc=None, quant_size=0.04):
    '''
    Visualise Tracking via object idx, scan as pointcloud for background, objects as voxel grids
    Added smoothing:
    - fused object shape in canonical space and averaged box coordinates
    '''

    boxes = []
    tracklets = dict()
    fused_shapes, fused_scales = fuse_obj_shape(pred_trajectories)

    for scan_idx in range(seq_len):
        for color_idx, traj in enumerate(pred_trajectories):

            if pc is not None:
                world_pc = pc

            norm_obj_shape = fused_shapes[color_idx]

            norm_obj_scale = fused_scales[color_idx]

            for frame in traj:
                if frame['scan_idx'] == scan_idx:

                    cad2scan = frame['obj']['pred_aligned2scan']

                    scan2world = np.identity(4)
                    scan2world[:3, :3] = np.diag([quant_size, quant_size, quant_size])

                    cad2world = scan2world @ cad2scan
                    #bx = frame['obj']['bbox'].astype(float) * quant_size
                    bx = norm_obj_scale.astype(float) * quant_size
                    s_x, s_y, s_z = bx[3] - bx[0], bx[4] - bx[1], bx[5] - bx[2]
                    obj_scale = np.array([s_x, s_y, s_z])

                    cad2world = rescale_mat(cad2world, obj_scale)
                    world_pc_obj = grid2world(norm_obj_shape, cad2world, None, pred=True)

                    if as_mesh:
                        mesh = vox2mesh(frame['obj']['occ'], box=None)

                        # Place at frame 0
                        if scan_idx == 0:  # todo if object is not in scan idx ==0 visible object is not placed maybe find all unique objects first
                            print('Placing {} in scene'.format(f'mesh_{color_idx}'))
                            dvis(mesh, fmt='mesh', c=color_idx+1, l=[0, 1], name=f'obj/mesh_{color_idx}')
                            # Timeout for loading object in dvis
                            time.sleep(5)
                        dvis(cad2world, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx+1)

                    elif scan_idx == 0:
                        dvis(world_pc_obj, vs=0.04, c=color_idx+1, t=scan_idx+1, l=[0,1], name=f'obj/{color_idx}')

                    # Bounding box placement
                    if with_box:
                        box = np.expand_dims(frame['obj']['bbox'].astype(float)*quant_size, axis=0)
                        dvis(box, fmt='box', s=3, c=color_idx+1, t=scan_idx+1, l=[0,3], name=f'box/box_{color_idx}')

                    # Obj pc center
                    obj_center = world_pc_obj.mean(axis=0)
                    if color_idx+1 in tracklets:
                        tracklets[color_idx+1].append(obj_center)
                    else:
                        tracklets[color_idx+1] = [obj_center]

                    # Boxes for cropping
                    if scan_idx == 0:
                        boxes.append(frame['obj']['bbox']*quant_size)
                    break

    # Set tracklet lines
    for c_val, l_verts in tracklets.items():
        line_verts = np.concatenate(l_verts, axis=0)
        dvis(line_verts, fmt='line', s=6, c=c_val, l=[0,2], name=f'line/{c_val}')

    # Vis background
    world_pc = crop_pc(world_pc, boxes)
    dvis(world_pc, fmt='xyzrgb', vs=0.02, l=[0,4], vis_conf={'opacity': 0.5}, name='background') #set opacity to 0.5

    # Set title
    dvis({"title":seq_name, "track_store_path": seq_name}, 'config')
    sys.exit()

    #Load and set camera parameters
    #dvis({}, fmt='cam')

## HELPER FCTS ---------------------------------------------------------------------------------------------------------
def grid2world(voxel_grid, cad2world, box, pred=False):
    '''
    Transform voxel obj to world space
    '''

    if not pred:
        if type(voxel_grid) == np.ndarray:
            nonzero_inds = np.nonzero(torch.from_numpy(voxel_grid))[:-1]
        else:
            nonzero_inds = np.nonzero(voxel_grid)[:-1]
    else:
        if type(voxel_grid) == np.ndarray:
            nonzero_inds = np.nonzero(torch.from_numpy(voxel_grid))[:-1]
        else:
            nonzero_inds = np.nonzero(voxel_grid)[:-1]

    max_dim = torch.max(nonzero_inds)
    points = nonzero_inds / max_dim - 0.5
    if points.is_cuda:
        points = points.detach().cpu().numpy()
    else:
        points = points.numpy()
    #points[:, 1] -= points[:, 1].min()  # CAD space y is shifted up to start at 0

    # Cad2World
    world_pc = cad2world[:3,:3] @ points.transpose() + np.expand_dims(cad2world[:3,3], axis=-1)
    world_pc = world_pc.T

    scaled_pc = world_pc
    if box is not None:
        scaled_pc[:,0] = minmax_scale(scaled_pc[:, 0], feature_range=(box[0], box[3]))
        scaled_pc[:,1] = minmax_scale(scaled_pc[:, 1], feature_range=(box[1], box[4]))
        scaled_pc[:,2] = minmax_scale(scaled_pc[:, 2], feature_range=(box[2], box[5]))

    return scaled_pc

def crop_pc(pc, boxes):
    '''
    Crop points which voxels will be placed at
    '''
    idxs = []
    for keep_idx, pt in enumerate(pc):
        keep = True
        for box in boxes:
            if pt[0] >= box[0]-0.01 and pt[0] <= box[3]+0.01 and pt[1] >= box[1]-0.01 and pt[1] <= box[4]+0.01 and pt[2] >= box[2]-0.01 and pt[2] <= box[5]+0.01:
                keep = False
        if keep == True:
            idxs.append(keep_idx)

    cropped_pc = pc[idxs,:]

    return cropped_pc

def prednorm_vox2mesh(vox, cad2world, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''
    if type(vox) == torch.Tensor:
        vox = vox.numpy()
    vertices, triangles = mcubes.marching_cubes(vox, 0)
    # Verticies to CAD space before applying transform
    vertices = vertices / 31 - 0.5

    scale = get_scale(cad2world[:3, :3])
    scale_mat = torch.diag(scale)

    world2cad = np.linalg.inv(cad2world.numpy())
    box = world2cad @ box_pts

    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.apply_transform(cad2world)

    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh


def norm_vox2mesh(vox, cad2world, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''
    if type(vox) == torch.Tensor:
        vox = vox.numpy()
    vertices, triangles = mcubes.marching_cubes(vox, 0)
    # Verticies to CAD space before applying transform
    vertices = vertices / 31 - 0.5

    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.apply_transform(cad2world)

    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh

def vox2mesh(vox, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''
    if type(vox) == torch.Tensor:
        vox = vox.numpy()
    vertices, triangles = mcubes.marching_cubes(vox, 0) # object space scan size
    dimensions = vox.shape
    #max_dim = max(dimensions[0], dimensions[1], dimensions[2]) - 1
    max_dim = vertices.max(axis=0)
    vertices = vertices / max_dim #- 0.5 # 0-centered
    #shift_y = vertices.min(axis=0)[1]
    #vertices[:,1] += shift_y # CAD space with scale 1
    # R_NEGY90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R_X90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rot_obj2cad = np.linalg.inv(R_X90)  # object space to cad space

    v_rot = rot_obj2cad @ vertices.transpose() + np.expand_dims(np.array([-0.5, 0, 0.5]), axis=-1)
    vertices = v_rot.transpose()
    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)


    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh


def fuse_obj_shape(pred_trajectories):
    '''
    fuse object shape by averaging over all predictions
    fuse object scale by averaging over all predictions
    '''

    fused_shapes = [[] for i in range(len(pred_trajectories))]
    fused_scales = [[] for i in range(len(pred_trajectories))]
    for traj_idx, traj in enumerate(pred_trajectories):
        shape = []
        scale = []
        max_dims = [0, 0, 0]
        for f_pred in traj:
            #dvis(f_pred['obj']['occ'], fmt='voxels')
            shape.append(f_pred['obj']['occ'].astype(int))
            #print('Scale', get_scale(f_pred['obj']['pred_aligned2scan'][:3,:3]))
            #scale.append(np.expand_dims(get_scale(f_pred['obj']['pred_aligned2scan'][:3,:3]), axis=0))
            scale.append(np.expand_dims(f_pred['obj']['bbox'], axis=0))

            # Get max dims
            x, y, z = f_pred['obj']['occ'].shape[0], f_pred['obj']['occ'].shape[1], f_pred['obj']['occ'].shape[2]
            max_dims[0] = max(max_dims[0], x)
            max_dims[1] = max(max_dims[1], y)
            max_dims[2] = max(max_dims[2], z)

        # Padding zeros to normalized dimension
        padded_shapes = []
        for sh in shape:
            pad_diff = np.array(max_dims) - sh.shape
            paddings = np.max([pad_diff, np.zeros(3)], 0).astype(int)
            p2d = [(0, i) for i in paddings]
            padded = np.pad(sh, p2d, "constant", constant_values=3 * [(0, 0)])
            padded_shapes.append(np.expand_dims(padded, axis=0))

        p_shapes = np.concatenate(padded_shapes, axis=0)
        shape = p_shapes.mean(axis=0)
        f_scales = np.concatenate(scale, axis=0).mean(axis=0)

        # Binarize shape again
        shape[shape >= 0.5] = 1
        shape[shape < 0.5] = 0
        fused_shapes[traj_idx] = shape
        fused_scales[traj_idx] = f_scales

    return fused_shapes, fused_scales

def fuse_obj_shape_gt(pred_trajectories, grid):
    '''
    fuse object shape by averaging over all predictions
    fuse object scale by averaging over all predictions
    '''

    fused_shapes = [[] for i in range(len(pred_trajectories))]
    fused_scales = [[] for i in range(len(pred_trajectories))]
    for traj_idx, traj in enumerate(pred_trajectories):
        for f_pred in traj:
            vg_crop(grid, box)

        fused_shapes[traj_idx] = shape
        fused_scales[traj_idx] = f_scales

    return fused_shapes, fused_scales

def rescale_mat(cad2world, norm_scale):
    '''
    Rescale cad2world matrix with fused scale parameter
    '''
    rot = cad2world[:3,:3]
    unscaled_rot = rot / get_scale(rot)
    scaled_rot = np.diag(norm_scale) @ unscaled_rot
    cad2world[:3, :3] = scaled_rot
    return cad2world

def unscale_mat(cad2world):
    '''
    Unscale cad2world matrix
    '''
    c2w_cpy = torch.clone(cad2world)
    rot = cad2world[:3,:3]
    scale = get_scale(rot)
    unscaled_rot = rot / scale
    c2w_cpy[:3, :3] = unscaled_rot
    return c2w_cpy

def get_scale(m):
    if type(m) == torch.Tensor:
        return m.norm(dim=0)
    return np.linalg.norm(m, axis=0)