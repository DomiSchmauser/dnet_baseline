import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import mathutils

from Tracking.utils.train_utils import convert_voxel_to_pc
import open3d as o3d


def visualize_graph(G, color):
    '''
    Visualise Graph data connectivity
    '''
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def visualise_tracking(input, output, use_gt_pose=False):
    '''
    Visualise Tracking output by colorizing every instance and placing all objects in
    one image for a whole sequence
    input: per sequence input with n-images
    output: predicted object associations for coloring instances
    use_gt_pose: use GT or predicted pose to place object for visualization
    '''

    pred_visobjects = []
    vis_idxs = output['vis_idxs'] # indicies for visualization, same len as output
    predictions = output['prediction']

    # Binarize predictions
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    if 'consecutive_mask' in output.keys():
        consecutive_mask = output['consecutive_mask']
        forward_connections = len(consecutive_mask)
        predictions = predictions[:forward_connections][consecutive_mask == 1]

    #targets = output['target']
    annotations = [None] * len(input) # list len images
    color_encodings = [[0,0.3,0.5 ], [0,0.4,0], [0.3,0,0.6 ], [0.6,0.6,0.6], [0.3,0.7,0.7], [0.4,0,0.6], [0.5,0.3,0.3], [1, 0.706, 0], [0, 0.651, 0.929]]


    # Create Mapping gt id to color encoding index
    gt_ids = []
    for v_id, vis_idx in enumerate(vis_idxs):
        if vis_idx['obj_id_1'] is not None:
            gt_ids.append(int(vis_idx['obj_id_1']))#
        if vis_idx['obj_id_2'] is not None:
            gt_ids.append(int(vis_idx['obj_id_2']))

    unique_ids = list(set(gt_ids))
    color_mapping = {}
    for loop_idx, u_id in enumerate(unique_ids):
        color_mapping[str(u_id)] = color_encodings[loop_idx]

    # Chaining of instances with positive connections
    for v_id, vis_idx in enumerate(vis_idxs):
        if predictions[v_id] == 1 and annotations[vis_idx['image']] is None:
            annotations[vis_idx['image']] = [vis_idx]
        elif predictions[v_id] == 1:
            annotations[vis_idx['image']].append(vis_idx)

        # add objects having no partner in consecutive image but still detected
        elif predictions[v_id] == 0 and v_id != len(input)-1:
            obj_id = vis_idx['obj_id_1']
            img2_idx = vis_idx['image'] + 1
            compare_ids = input[img2_idx]['gt_object_id']

            if obj_id not in compare_ids:
                if annotations[vis_idx['image']] is None:
                    annotations[vis_idx['image']] = [vis_idx]
                else:
                    annotations[vis_idx['image']].append(vis_idx)

    full_drawn_objects = []

    for idx, img in enumerate(input):

        if idx == 0:
            print('Scene :', img['scene'])

        num_instances = img['voxels'].shape[0]
        img_colors = [None] * num_instances

        if idx != len(input)-1: # skip last
            num_instances_2 = input[idx+1]['voxels'].shape[0]
            img_colors_2 = [None] * num_instances_2

        img_annos = annotations[idx]

        if img_annos is not None: #  last frame always None -> uses img_colors_2 from previous frame

            for ann_idx, ann in enumerate(img_annos):
                idx_1 = int(ann['obj_1'])
                idx_2 = int(ann['obj_2'])
                key_1 = str(ann['obj_id_1'])
                key_2 = str(ann['obj_id_2'])

                if ann['obj_id_1'] is not None:
                    img_colors[idx_1] = color_mapping[key_1]
                else:
                    img_colors[idx_1] = [1, 0, 0]
                if ann['obj_id_2'] is not None: # Only for last frame
                    img_colors_2[idx_2] = color_mapping[key_2]
                else:
                    img_colors_2[idx_2] = [1, 0, 0]

        for i in range(num_instances):

            voxel_grid = img['voxels'][i,:,:,:]
            scale = img['scales'][i].numpy()
            if use_gt_pose:
                euler_rot = img['gt_rotations'][i].numpy()
                translation = img['gt_locations'][i].numpy()
            else:
                euler_rot = img['rotations'][i].numpy()
                translation = img['translations'][i].numpy()
            rotation = np.array(mathutils.Euler((euler_rot)).to_matrix())

            world_pc = convert_voxel_to_pc(voxel_grid, rotation, translation, scale) # use scale since uses GT
            world_pc_obj = o3d.geometry.PointCloud()

            obj_color = img_colors[i]

            if obj_color is not None and obj_color not in full_drawn_objects:
                world_pc_obj.points = o3d.utility.Vector3dVector(world_pc)
                every_k_pts = 2
                world_pc_obj = world_pc_obj.uniform_down_sample(every_k_pts)

                if obj_color is not None and idx != len(input) - 1:
                    world_pc_obj.paint_uniform_color(obj_color)
                    full_drawn_objects.append(obj_color)
                elif idx == len(input) - 1 and img_colors_2[i]:
                    world_pc_obj.paint_uniform_color(img_colors_2[i])
                else:
                    world_pc_obj.paint_uniform_color([1, 0, 0])  # Red for wrong predictions

                pred_visobjects.append(world_pc_obj)

            else:
                obj_box = o3d.geometry.OrientedBoundingBox()
                obj_box = obj_box.create_from_points(o3d.utility.Vector3dVector(world_pc))
                corner_pts = np.asarray(obj_box.get_box_points())

                # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
                lines = [[0, 2], [1, 6], [2, 5], [0, 3],
                         [3, 6], [4, 5], [4, 6], [4, 7],
                         [0, 1], [1, 7], [2, 7], [3, 5]]

                # Use the same color for all lines
                if obj_color is not None and idx != len(input) - 1:
                    colors = [obj_color for _ in range(len(lines))]
                    full_drawn_objects.append(obj_color)
                elif idx == len(input) - 1 and img_colors_2[i]:
                    colors = [img_colors_2[i] for _ in range(len(lines))]
                else:
                    colors = [[1, 0, 0] for _ in range(len(lines))]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corner_pts)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                pred_visobjects.append(line_set)

    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pred_visobjects.append(nocs_origin)
    o3d.visualization.draw_geometries(pred_visobjects)