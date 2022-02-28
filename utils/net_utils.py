import numpy as np
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 downsample=None
                 ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.norm1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.norm2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def iou3d(boxes, query_boxes):
    box_ares = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])
    query_ares = (
        (query_boxes[:, 3] - query_boxes[:, 0])
        * (query_boxes[:, 4] - query_boxes[:, 1])
        * (query_boxes[:, 5] - query_boxes[:, 2])
    )

    iw = (torch.min(boxes[:, 3], query_boxes[:, 3]) - torch.max(boxes[:, 0], query_boxes[:, 0])).clamp(min=0)
    ih = (torch.min(boxes[:, 4], query_boxes[:, 4]) - torch.max(boxes[:, 1], query_boxes[:, 1])).clamp(min=0)
    il = (torch.min(boxes[:, 5], query_boxes[:, 5]) - torch.max(boxes[:, 2], query_boxes[:, 2])).clamp(min=0)
    ua = box_ares+ query_ares - iw * ih * il 
    overlaps = iw * ih * il / ua
    return overlaps


def cats(source, target, dim=0):
    
    if source is None or len(source) == 0:
        return target
    if target is None or len(target) == 0:
        return source
    
    if type(source) == np.ndarray:
        return np.concatenate([source, target], axis=dim)
    
    return torch.cat([source, target], dim=dim)

def flatten_lists(l, detach=False):
    if type(l) == list:
        if len(l) > 0:
            if type(l[0]) == torch.Tensor:
                if detach:
                    return [np.mean(x.data.cpu().numpy()) for x in l]
                else:
                    return [x for x in l]
            elif type(l[0]) == list:
                return [y for x in l for y in flatten_lists(x, detach)]
            else:
                return [x for x in l]
    return l



def merge_dict(a, b, detach=False):

    for key, val in b.items():
        if type(val) == dict:
            a[key] = merge_dict(a.get(key, dict()), val, detach)
        elif type(val) == torch.Tensor:
            if detach:
                a[key] = a.get(key, []) + [torch.mean(val).item()]
            else:
                a[key] = a.get(key, []) + [val]
        elif type(val) == float:
            a[key] = a.get(key, []) + [val]
        elif type(val) == list:
            a[key] = a.get(key, []) + flatten_lists(val, detach)
            """
            if detach:
                if type(val[0]) == torch.Tensor:
                    a[key] = a.get(key, []) + [x.data.cpu().numpy() for x in val]
                elif type(val[0]) == list:
                    a[key] = a.get(key, []) + [x.data.cpu().numpy() for x in val]
                else:
                   a[key] = a.get(key, []) + val
            """
    return a


def vg_add_crop(vg, vg2, min_coords, spatial_end=True):

    vg_added = vg
    if spatial_end:
        bbox = vg2.shape[-3:]
        vg_added[..., int(min_coords[0]):int(bbox[0]), int(min_coords[1]):int(bbox[1]), int(min_coords[2]):int(bbox[2])] += vg2
    
    return vg_added



def insec_vg_crop(vgs, vg_bboxes, bboxes, spatial_end=True):
    # vgs: List[FWLH]
    # vg_mins: List[ xyz]
    # bboxes: Nx6
    # targets: N indices
    insec_vgs = []
    insec_bboxes = []
    for vg, vg_bbox, bbox in zip(vgs, vg_bboxes, bboxes):
        insec_bbox = torch.cat([torch.max(vg_bbox[:3], bbox[:3].int()), torch.min(vg_bbox[3:6], bbox[3:6].int())])
        rel_bbox = torch.cat([torch.max(vg_bbox[:3], bbox[:3].int()) - vg_bbox[:3], torch.min(vg_bbox[3:6], bbox[3:6].int()) - vg_bbox[:3]])
        # the intersection bbox relative to vg coordinates
        insec_bboxes.append(insec_bbox)
        if torch.any(rel_bbox[:3] + 3 > rel_bbox[3:6]):
            insec_vgs.append(None)
        else:
            insec_vgs.append(vg_crop(vg, rel_bbox, spatial_end))
            
    return insec_vgs, insec_bboxes

def vg_crop(vg, bboxes, spatial_end=True, crop_box=False):
    # vg: ... X W X L X H
    # bboxes: N x (min, max,...) or (min,max,...)
    if type(vg) == np.ndarray:
        return vg_crop_np(vg, bboxes, spatial_end=spatial_end, crop_box=crop_box)

    if len(bboxes.shape) == 1:
        if spatial_end:
            if not crop_box:
                assert torch.all(bboxes[:3] >= 0) and torch.all(bboxes[3:6] <= torch.tensor(vg.shape[-3:]).to(bboxes.device) )
                return vg[..., int(bboxes[0]):int(bboxes[3]), int(bboxes[1]):int(bboxes[4]), int(bboxes[2]):int(bboxes[5])]
            else:
                bbox_cropped = torch.cat([torch.max(bboxes[:3], torch.zeros(3).to(bboxes.device).type_as(bboxes)),
                torch.min(bboxes[3:],torch.Tensor([*vg.shape[-3:]]).to(bboxes.device).type_as(bboxes))
                ], 0)
                return vg[
                    ...,
                    int(bbox_cropped[0]) : int(bbox_cropped[3]),
                    int(bbox_cropped[1]) : int(bbox_cropped[4]),
                    int(bbox_cropped[2]) : int(bbox_cropped[5]),
                ]
        else:
            if not crop_box:
                assert torch.all(bboxes[:3] >= 0) and torch.all(bboxes[3:6] <= torch.tensor(vg.shape[:3]).to(bboxes.device) )
                return vg[int(bboxes[0]) : int(bboxes[3]), int(bboxes[1]) : int(bboxes[4]), int(bboxes[2]) : int(bboxes[5])]
            else:
                bbox_cropped = torch.cat([torch.max(bboxes[:3], torch.zeros(3).to(bboxes.device).type_as(bboxes)),
                torch.min(bboxes[3:],torch.Tensor([*vg.shape[:3]]).to(bboxes.device).type_as(bboxes))
                ], 0)
                
                return vg[
                    int(bbox_cropped[0]) : int(bbox_cropped[3]),
                    int(bbox_cropped[1]) : int(bbox_cropped[4]),
                    int(bbox_cropped[2]) : int(bbox_cropped[5]),
                ]



    if len(bboxes.shape) == 2:
        return [ vg_crop(vg, bbox, spatial_end) for bbox in bboxes]



def vg_crop_np(vg, bboxes, spatial_end=True, crop_box=False):
    # vg: ... X W X L X H
    # bboxes: N x (min, max,...) or (min,max,...)
    if len(bboxes.shape) == 1:
        if spatial_end:
            if not crop_box:
                assert np.all(bboxes[:3] >= 0) and np.all(bboxes[3:6] <= vg.shape[-3:])
                return vg[..., int(bboxes[0]) : int(bboxes[3]), int(bboxes[1]) : int(bboxes[4]), int(bboxes[2]) : int(bboxes[5])]
            else:
                bbox_cropped = np.concatenate([np.max([bboxes[:3], np.zeros(3)], 0), np.min([bboxes[3:], vg.shape[-3:]], 0)], 0)
                return vg[
                    ...,
                    int(bbox_cropped[0]) : int(bbox_cropped[3]),
                    int(bbox_cropped[1]) : int(bbox_cropped[4]),
                    int(bbox_cropped[2]) : int(bbox_cropped[5]),
                ]
        else:
            if not crop_box:
                assert np.all(bboxes[:3] >= 0) and np.all(bboxes[3:6] <= vg.shape[:3])
                return vg[int(bboxes[0]) : int(bboxes[3]), int(bboxes[1]) : int(bboxes[4]), int(bboxes[2]) : int(bboxes[5])]
            else:
                bbox_cropped = np.concatenate([np.max([bboxes[:3], np.zeros(3)], 0), np.min([bboxes[3:], vg.shape[:3]], 0)], 0)
                return vg[
                    int(bbox_cropped[0]) : int(bbox_cropped[3]),
                    int(bbox_cropped[1]) : int(bbox_cropped[4]),
                    int(bbox_cropped[2]) : int(bbox_cropped[5]),
                ]
    if len(bboxes.shape) == 2:
        return [vg_crop(vg, bbox, spatial_end, crop_box) for bbox in bboxes]




def bbox_reshape(bbox, lvl_in, dec_lvl_out):
    bbox_rs = bbox
    for scale_factor in cfg.model.submodels.backbone.net.encoder_lvl[lvl_in:dec_lvl_out+1]:
        if type(bbox) == torch.Tensor:
            bbox_rs = (bbox_rs / scale_factor).int()
        else:
            bbox_rs = (bbox_rs / scale_factor).astype(int)
    return bbox_rs

def hmgt(arr):
    if len(arr.shape) == 2:
        if arr.shape[1] == 4:
            extended = arr
            extended[:, 3] = 1
        elif arr.shape[1] == 3 and arr.shape[0] == 3:
            extended = torch.eye(4).type_as(arr)
            extended[:3,:3] = arr
        else:
            extended = torch.ones(arr.shape[0], arr.shape[1] + 1).type_as(arr)
            extended[:, :3] = arr
        return extended
    else:
        extended = torch.ones(4).type_as(arr)
        extended[:3] = arr
        return extended


def hmg(arr):
    if len(arr.shape) == 2:
        extended = np.ones((arr.shape[0], arr.shape[1] + 1))
        extended[:, :3] = arr
        return extended
    else:
        return np.array([*arr[:3], 1])



def get_scale(m):
    if type(m) == torch.Tensor:
        return m.norm(dim=0)
    return np.linalg.norm(m, axis=0)


def kabsch_rot(P_cent, Q_cent):
    if P_cent.shape[1] < 6:
        return torch.eye(3).type_as(P_cent)

    # P_cent, Q_cent <3,n>
    H = P_cent @ Q_cent.t()
    U, S, V = torch.svd(H.double())
    if (S.abs().max() / S.abs().min()) > 1000:
        return torch.eye(3).type_as(P_cent)
    d = torch.det(V @ U.t())
    R = V @ torch.diag(torch.Tensor([1, 1, torch.sign(d)]).type_as(P_cent).double()) @ U.t()

    # kabsch_rot2(P_cent, Q_cent)
    return R

def kabsch_trans_rot(P, Q):
    # P, Q <3,n>
    P = P.double()
    Q = Q.double()
    p_0 = P.mean(1)
    q_0 = Q.mean(1)
    P_cent = (P.t() - p_0).t()
    Q_cent = (Q.t() - q_0).t()
    R = kabsch_rot(P_cent, Q_cent)
    T = q_0 - R @ p_0
    return T.float(), R.float()


def get_aligned2noc():
    aligned2noc = torch.diag(torch.Tensor([0.5, 0.5, -0.5, 1]))
    aligned2noc[:3,3] = torch.Tensor([0.5,0.5,.5])
    return aligned2noc



def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x3 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 3)  # Nx3x3
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x3 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx3
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a N x 3 x 3  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~ mask_d2) * (~ mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4