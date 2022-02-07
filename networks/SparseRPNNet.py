import torch
from torch import nn
import torch.nn.functional as F
from utils.ME_utils import BasicBlock

import MinkowskiEngine as ME



class SparseRPNNet4_Res1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf['num_features']


        self.enc1 = nn.Sequential(
            BasicBlock(self.num_features*2, self.num_features*2, dimension=3),
            BasicBlock(self.num_features* 2, self.num_features* 2, dimension=3),
            ME.MinkowskiConvolution(self.num_features *2 , self.num_features  , kernel_size=1, stride=1, dimension=3),
        )
        self.reg = nn.Sequential(
            BasicBlock(self.num_features, self.num_features, dimension=3),
            ME.MinkowskiConvolution(self.num_features  , 8  , kernel_size=1, stride=1, dimension=3),
            
        )
        
    def forward(self, x):
        # Encode
        x_e1 = self.enc1(x)
        reg_vals = self.reg(x_e1)
        return reg_vals



class SparseRPNNet4_Unet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.nf= conf.num_features

        ch = [self.nf//2, self.nf, 2*self.nf, 4*self.nf, 2**3*self.nf, 2**4*self.nf]


        self.enc1 = nn.Sequential(
            BasicBlock(2*ch[2], 2*ch[2], dimension=3),
            BasicBlock(2*ch[2], 2*ch[2], dimension=3),
            ME.MinkowskiConvolution(2*ch[2] , ch[1] , kernel_size=1, stride=1, dimension=3),
        )
        self.reg = nn.Sequential(
            BasicBlock(ch[1], ch[1], dimension=3),
            ME.MinkowskiConvolution(ch[1]  , 8  , kernel_size=1, stride=1, dimension=3),
            
        )
        
        
    def forward(self, x):
        # Encode
        x_e1 = self.enc1(x)
        reg_vals = self.reg(x_e1)
        return reg_vals



class SparseRPNNet4(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features
        self.num_anchors = conf.num_anchors

        self.enc1 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 8, self.num_features*4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features*4),
            ME.MinkowskiReLU(),
        )
        self.reg = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features *4 , self.num_features  , kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiConvolution(self.num_features  , 8  , kernel_size=1, stride=1, dimension=3),
            
        )
        
        
    def forward(self, x):
        # Encode
        x_e1 = self.enc1(x)
        reg_vals = self.reg(x_e1)
        return reg_vals



class SparseRPNNet8(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features
        self.num_anchors = conf.num_anchors

        self.enc1 = nn.Sequential(
            nn.Conv3d(self.num_features * 8, self.num_features *2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features*2),
            nn.ReLU(),
            nn.Conv3d(self.num_features * 2, self.num_features , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU(),
        )
        self.objness_scores = nn.Sequential(
            nn.Conv3d(self.num_features, self.num_anchors * 2, kernel_size=1, stride=1, padding=0),
        )
        self.bbox_delta_regs = nn.Sequential(
            nn.Conv3d(self.num_features, self.num_anchors * 6, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, x):
        # Encode
        x_e1 = self.enc1(x)
        rpn_logits = self.objness_scores(x_e1)
        # Bx2*AxWxHxL -> BxWxHxLxAx2 -> BxWHLAx2

        rpn_logits = rpn_logits.permute(0,2,3,4,1).reshape(x.shape[0],-1,2)
        rpn_bboxes = self.bbox_delta_regs(x_e1)

        rpn_bboxes = rpn_bboxes.permute(0,2,3,4,1).reshape(x.shape[0], -1,6)
        return rpn_logits, rpn_bboxes
