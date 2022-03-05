import torch
import numpy as np
from trimesh.transformations import rotation_matrix, translation_matrix, scale_matrix, euler_from_matrix, decompose_matrix
from block_timer import timer
#from gesvd import GESVD


def hmgt(arr):
    if len(arr.shape) == 2:
        if arr.shape[1] == 4:
            extended = arr
            extended[:, 3] = 1
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
        if arr.shape[1] == 4:
            extended = arr
            extended[:, 3] = 1
        else:
            extended = np.ones((arr.shape[0], arr.shape[1] + 1))
            extended[:, :3] = arr
        return extended
    else:
        extended = np.ones(4).astype(arr)
        extended[:3] = arr
        return extended

def ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(Sxy)
    #d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.linalg.det(Sxy) < 0 ):
            S[m, m] = -1
        elif (r == m ):
            if (np.linalg.det(U) * np.linalg.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(m)
            c = 1
            t = np.zeros(m)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R,c,t


def umeyama(from_points, to_points):
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    
    N, m = from_points.shape
    
    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)
    
    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m
    
    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
    
    cov_matrix = delta_to.T.dot(delta_from) / N
    
    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)
    
    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        #raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        #return np.eye(3), 1, mean_to
        # S[m, m] = -1
        pass
    
    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c*R.dot(mean_from)
    
    return R,c,t



def umeyama_torch(from_points, to_points):
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    
    N, m = from_points.shape
    
    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)
    
    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m
    
    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
    
    cov_matrix = delta_to.T @ (delta_from) / N
    #with timer.Timer('svd'):
        #svd = GESVD()
    U, d, V = torch.svd(cov_matrix) #svd(cov_matrix) #
    V_t = V.T
    cov_rank = torch.matrix_rank(cov_matrix)
    S = torch.eye(m).to(from_points)
    
    if cov_rank >= m - 1 and torch.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        #raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        return S, 1/sigma_from, mean_to - 1/sigma_from*mean_from

    
    R = U @ S @ V_t
    c = (d * S.diag()).sum() / sigma_from
    t = mean_to - (c*R) @ mean_from
    
    return R,c,t

def estimateSimilarityUmeyama(SourceHom, TargetHom):

    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    if varP * np.sum(D) != 0:
        ScaleFact = 1/varP * np.sum(D)  # scale factor
    else:
        ScaleFact = 1  # scale factor set to 1 since otherwise division by 0

    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    return torch.from_numpy(Rotation), torch.from_numpy(ScaleMatrix), torch.from_numpy(Translation)



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



def kabsch_trs(P,Q):
    m, n = P.shape
    P = P.double()
    Q = Q.double()
    p_0 = P.mean(1)
    q_0 = Q.mean(1)
    P_cent = (P.t() - p_0).t()
    Q_cent = (Q.t() - q_0).t()
    sp = torch.mean(torch.sum(P_cent*P_cent,0))
    sq = torch.mean(torch.sum(Q_cent*Q_cent,0))
    Spq = (sq @ sq.T) / n
    U, D, V = torch.svd(Spq)
    if (Spq.abs().max() / Spq.abs().min()) > 1000:
        return torch.eye(3).type_as(P_cent)
    r = torch.sum(torch.diag(Spq)>0)
    if r >= (m - 1):
        if torch.det(Spq) < 0:
            Spq[m,m] = -1
        elif r == m - 1:
            if torch.det(U) * np.det(V) < 0:
                Spq[m,m] = -1
        else:
            d  = torch.det(V @ U.t())
        
        S = torch.diag(torch.Tensor([1, 1, torch.sign(d)]).type_as(P_cent).double())

    R = V @ S @ U.t() # correct order??
    c = torch.trace(D @ S) / sp
    t = q_0 - c * R * p_0
        

def kabsch_trans_rot(P, Q):
    # P, Q <3,n>
    P = P.double()
    Q = Q.double()
    p_0 = P.mean(1)
    q_0 = Q.mean(1)
    P_cent = (P.t() - p_0).t()
    Q_cent = (Q.t() - q_0).t()
    sp = torch.mean(torch.sum(P_cent*P_cent,0))
    sq = torch.mean(torch.sum(Q_cent*Q_cent,0))
    Spq = (sq @ sq.T) / len(P)

    R = kabsch_rot(P_cent, Q_cent)
    T = q_0 - R @ p_0
    return T.float(), R.float()


if __name__ == '__main__':

    # Run an example test
    # We have 3 points in 3D. Every point is a column vector of this matrix A
    A=np.random.rand(3,2000) # np.array([[0.57215 ,  0.37512 ,  0.37551] ,[0.23318 ,  0.86846 ,  0.98642],[ 0.79969 ,  0.96778 ,  0.27493]])
    # Deep copy A to get B
    B= (translation_matrix([3,4,5]) @ rotation_matrix(0.3,[1,1,1]) @ scale_matrix(3.2) @ hmg(A.T).T).T[:,:3].T

    #A = np.array([[1,0,0],[1.1,0,0], [1.2,0,0]]).T
    #B= (rotation_matrix(np.pi,[0,1,0]) @ hmg(A.T).T).T[:,:3].T
    # Reconstruct the transformation with ralign.ralign

    A_torch = torch.from_numpy(A).cuda()
    B_torch = torch.from_numpy(B).cuda()



    with timer.Timer('np'):
        R, c, t = ralign(A, B)
    with timer.Timer('np2'):
        R2, c2, t2 = umeyama(A.T,B.T) 
    with timer.Timer('torch'):
        R_torch, c_torch, t_torch = umeyama_torch(A_torch.T, B_torch.T)
    print(R)
    print(c)
    print(t)
    print(euler_from_matrix(R))
    decompose_matrix(rotation_matrix(np.pi,[0,1,0]))
    assert np.allclose( R @ (c*A) +np.tile(t, (A.shape[1],1)).T, B )