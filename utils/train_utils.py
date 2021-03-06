from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from scipy.spatial import ConvexHull
from torch.nn import init

import sys

def loss_to_logging(overall_losses):

    log_losses = dict()
    norm_coeff_rpn = 0
    norm_coeff_comp = 0
    norm_coeff_noc = 0
    norm_coeff_total = 0

    for losses in overall_losses:
        for k, v in losses.items():

            if k == 'rpn':
                if 'rpn' not in log_losses:
                    log_losses[k] = 0
                for m in range(len(losses[k]['bweighted_loss'])):
                    log_losses[k] += v['bweighted_loss'][m].detach().cpu().item()
                    norm_coeff_rpn += 1

            elif k == 'completion':
                if 'completion' not in log_losses:
                    log_losses[k] = 0
                for m in range(len(losses[k]['bweighted_loss'])):
                    log_losses[k] += torch.mean(v['bweighted_loss'][m]).detach().cpu().item()
                    norm_coeff_comp += 1

            elif k == 'noc':
                if 'noc' not in log_losses:
                    log_losses[k] = 0
                for m in range(len(losses[k]['bweighted_loss'])):
                    log_losses[k] += torch.mean(v['bweighted_loss'][m]).detach().cpu().item()
                    norm_coeff_noc += 1

            else:
                norm_coeff_total += 1
                if k not in log_losses:
                    log_losses[k] = v
                else:
                    log_losses[k] += v

    # Normalize loss
    for k_l, v_l in log_losses.items():
        if k_l == 'rpn':
            log_losses[k_l] /= norm_coeff_rpn
        if k_l == 'completion':
            log_losses[k_l] /= norm_coeff_comp
        if k_l == 'noc':
            log_losses[k_l] /= norm_coeff_noc
        if k_l == 'total_loss':
            log_losses[k_l] /= norm_coeff_total

    return log_losses

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def check_pair(pred_bbox, gt_bboxes, gt_ids, thres=0.01):

    ious = []
    for i in range(gt_bboxes.shape[0]):
        iou, _ = compute_3d_iou(pred_bbox, gt_bboxes[i,:,:])
        ious.append(iou)

    max_iou = np.array(ious).max()
    max_iou_idx = np.argmax(np.array(ious))
    if max_iou >= thres:
        obj_id = gt_ids[max_iou_idx]
    else:
        obj_id = None

    return obj_id

def compute_3d_iou(corners1, corners2):

    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])

    inter_vol = inter_area * max(0.0, ymax - ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    return iou, iou_2d

# Helper functions --------------------------------------------

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def box3d_vol(corners):

    ''' corners: (8,3) no assumption on axis direction '''

    a = torch.sqrt(torch.sum((corners[0,:] - corners[1,:])**2))
    b = torch.sqrt(torch.sum((corners[1,:] - corners[2,:])**2))
    c = torch.sqrt(torch.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def convert_voxel_to_pc(voxel_grid, rot, trans, scale):
    '''
    Converts a voxel grid to a point cloud with according pose
    voxel_grid: 32x32x32 tensor binary
    rot, trans, scale: output from run pose function
    returns pc: n x 3 array
    '''

    nonzero_inds = np.nonzero(voxel_grid)[:-1]
    points = nonzero_inds / 32 - 0.5
    points = points.detach().cpu().numpy()

    global_scalerot = (np.identity(3) * scale.copy()) @ rot
    world_pc = global_scalerot @ points.transpose() + np.expand_dims(trans.copy(), axis=-1)
    world_pc = world_pc.transpose()

    return world_pc

def _sparsify(data, th):

    valid_mask = np.abs(data) <= th # signed distance truncated

    coords = np.argwhere(valid_mask)[:, [0, 2, 3, 4]] # mask out coords
    feats = np.expand_dims(data[valid_mask], 0).T
    # coords in format B, x y z

    return (
        torch.from_numpy(feats).float(),
        torch.from_numpy(coords).int(),
    )

