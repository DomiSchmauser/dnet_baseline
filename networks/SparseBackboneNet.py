import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME
#from dvis import dvis
from utils.ME_utils import BasicBlock

from block_timer.timer import Timer

class PureSparseBackboneEPND2_Ex1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(1, self.num_features, kernel_size=5, stride=2, dimension=3),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features, self.num_features , kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features, self.num_features , kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiReLU(),
        )
        self.conv2 = ME.MinkowskiConvolution(self.num_features, self.num_features * 2, kernel_size=3, stride=1, dimension=3)
        self.enc2 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features*2, self.num_features * 4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 4),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features*4, self.num_features * 8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 8),
            ME.MinkowskiReLU(),
        )
               
    def forward(self, x, c_out):
        # Encode
        x_e1 = self.enc1(x)
        x_c2 = self.conv2(x_e1, c_out.C)
        # aaa  = ME.MinkowskiConvolution(self.num_features*2, self.num_features*2, kernel_size=9, stride=1, dimension=3).cuda()
        # aaa._parameters['kernel'] = nn.Parameter(torch.ones_like(aaa._parameters['kernel'])).cuda()
        # m = ME.SparseTensor(torch.ones(2,self.num_features*2).cuda(), torch.Tensor([[0,0,16,28], [0,0,20,28]]).cuda(), force_creation=True)
        if False:
            m = ME.SparseTensor(torch.ones(2,self.num_features*2).cuda(), torch.Tensor([[0,0,10,28], [0,0,20,28]]).cuda(), force_creation=True)
            aaa3  = ME.MinkowskiConvolution(self.num_features*2, self.num_features*2, kernel_size=1, stride=1, dimension=3).cuda()
            aaa3._parameters['kernel'] = nn.Parameter(torch.ones_like(aaa3._parameters['kernel'])).cuda()
            m2 = aaa3(m)
            m2.set_tensor_stride([10,10,10])
            aaa2  = ME.MinkowskiConvolution(self.num_features*2, self.num_features*2, kernel_size=3, stride=1, dimension=3).cuda()
            aaa2._parameters['kernel'] = nn.Parameter(torch.ones_like(aaa2._parameters['kernel'])).cuda()
            r1 = aaa2(m2)
            mp = ME.SparseTensor(m2.F, torch.Tensor([[0,0,10,28], [0,0,20,28]]).cuda(), force_creation=True)
            mp.set_tensor_stride([10, 10, 10])
            r2 = aaa2(mp)
            # r1 == r2

        # this should work!
        x_c2.set_tensor_stride([4,4,4])
        x_e2 = self.enc2(x_c2)
        
        return x_e1, x_e2



class PureSparseBackboneCol_Ex1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(4, self.num_features, kernel_size=5, stride=2, dimension=3),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features, self.num_features , kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features, self.num_features , kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiReLU(),
        )
        self.conv2 = ME.MinkowskiConvolution(self.num_features, self.num_features * 2, kernel_size=3, stride=1, dimension=3)
        # TODO STRIDE USED AT KERNEL_SIZE?
        self.enc2 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features*2, self.num_features * 4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 4),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features*4, self.num_features * 8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 8),
            ME.MinkowskiReLU(),
        )
               
    def forward(self, x_geo, x_col, c_out):
        # Encode
        x = ME.cat(x_geo,x_col)
        x_e1 = self.enc1(x)
        x_c2 = self.conv2(x_e1, c_out.C)
        # aaa  = ME.MinkowskiConvolution(self.num_features*2, self.num_features*2, kernel_size=9, stride=1, dimension=3).cuda()
        # aaa._parameters['kernel'] = nn.Parameter(torch.ones_like(aaa._parameters['kernel'])).cuda()
        # m = ME.SparseTensor(torch.ones(2,self.num_features*2).cuda(), torch.Tensor([[0,0,16,28], [0,0,20,28]]).cuda(), force_creation=True)
        if False:
            m = ME.SparseTensor(torch.ones(2,self.num_features*2).cuda(), torch.Tensor([[0,0,10,28], [0,0,20,28]]).cuda(), force_creation=True)
            aaa3  = ME.MinkowskiConvolution(self.num_features*2, self.num_features*2, kernel_size=1, stride=1, dimension=3).cuda()
            aaa3._parameters['kernel'] = nn.Parameter(torch.ones_like(aaa3._parameters['kernel'])).cuda()
            m2 = aaa3(m)
            m2.set_tensor_stride([10,10,10])
            aaa2  = ME.MinkowskiConvolution(self.num_features*2, self.num_features*2, kernel_size=3, stride=1, dimension=3).cuda()
            aaa2._parameters['kernel'] = nn.Parameter(torch.ones_like(aaa2._parameters['kernel'])).cuda()
            r1 = aaa2(m2)
            mp = ME.SparseTensor(m2.F, torch.Tensor([[0,0,10,28], [0,0,20,28]]).cuda(), force_creation=True)
            mp.set_tensor_stride([10, 10, 10])
            r2 = aaa2(mp)
            # r1 == r2

        # this should work!
        x_c2.set_tensor_stride([4,4,4])
        x_e2 = self.enc2(x_c2)
        
        return x_e1, x_e2


class PureSparseBackboneGeo_Res1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(1, self.num_features, kernel_size=5, stride=2, dimension=3),
            ME.MinkowskiReLU(),
            BasicBlock(self.num_features, self.num_features, dimension=3),
            BasicBlock(self.num_features, self.num_features, dimension=3),
        )
        self.conv2 = ME.MinkowskiConvolution(self.num_features, self.num_features * 2, kernel_size=3, stride=1, dimension=3)
        self.enc2 = nn.Sequential(
            BasicBlock(self.num_features*2, self.num_features*2, dimension=3),
            BasicBlock(self.num_features*2, self.num_features*2, dimension=3),
        )
               
    def forward(self, x_geo, c_out):
        # Encode
        x = x_geo
        x_e1 = self.enc1(x)
        x_c2 = self.conv2(x_e1, c_out.C)

        # this should work!
        x_c2.set_tensor_stride([4,4,4])
        x_e2 = self.enc2(x_c2)
        
        return x_e2



class PureSparseBackboneCol_Res1(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.num_features = conf['num_features']

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(3, self.num_features, kernel_size=5, stride=2, dimension=3), # ORIGINAL = 4
            ME.MinkowskiReLU(),
            BasicBlock(self.num_features, self.num_features, dimension=3),
            BasicBlock(self.num_features, self.num_features, dimension=3),
        )
        self.conv2 = ME.MinkowskiConvolution(self.num_features, self.num_features * 2, kernel_size=3, stride=1, dimension=3)
        self.enc2 = nn.Sequential(
            BasicBlock(self.num_features*2, self.num_features*2, dimension=3),
            BasicBlock(self.num_features*2, self.num_features*2, dimension=3),
        )
               
    def forward(self, x_geo, x_col, c_out):
        # Encode
        #x = ME.cat(x_geo,x_col)
        x = x_geo
        x_e1 = self.enc1(x)
        x_c2 = self.conv2(x_e1, c_out.C)

        # this should work!
        x_c2.set_tensor_stride([4,4,4])
        x_e2 = self.enc2(x_c2)
        
        return x_e2


class SparseBackboneEPND2_Ex1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(1, self.num_features, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiReLU(),
        )
        self.enc2 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features, self.num_features , kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features, self.num_features , kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.num_features, self.num_features * 2, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 2),
            ME.MinkowskiReLU(),

        )
        self.enc3 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 2, self.num_features * 4, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 4),
            ME.MinkowskiReLU(),
        )
        self.enc4 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 4, self.num_features * 8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 8),
            ME.MinkowskiReLU(),
        )

        # int(( c + 2*p-(k-1)-1) / stride+ 1)

        self.bottleneck = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 8, self.num_features * 8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 8),
            ME.MinkowskiReLU(),
        )

        self.dec4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(2 * self.num_features * 8, self.num_features * 4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 4),
            ME.MinkowskiReLU(),
        )

        # (c-1)*s - 2*p + (k-1)+1 +op = (c-1)*s -2*p+k+op 
        self.dec3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(2 * self.num_features * 4, self.num_features * 2, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(2 * self.num_features),
            ME.MinkowskiReLU(),
        )
       
    def forward(self, x):
        # Encode
        x_e1 = self.enc1(x)
        x_e2 = self.enc2(x_e1)
        x_e3 = self.enc3(x_e2)
        x_e4 = self.enc4(x_e3)
        # Bottleneck
        x_d4 = self.bottleneck(x_e4)
        # Decode
        x_d3 = self.dec4(ME.cat(x_d4, x_e4 ))
        x_d2 = self.dec3(ME.cat(x_d3, x_e3))
        
        return x_e1, x_e2, x_e3, x_e4, x_d4, x_d3, x_d2


class SparseBackboneEPND2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(1, self.num_features, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiReLU(),
        )
        self.enc2 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features, self.num_features * 2, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 2),
            ME.MinkowskiReLU(),

        )
        self.enc3 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 2, self.num_features * 4, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 4),
            ME.MinkowskiReLU(),
        )
        self.enc4 = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 4, self.num_features * 8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 8),
            ME.MinkowskiReLU(),
        )

        # int(( c + 2*p-(k-1)-1) / stride+ 1)

        self.bottleneck = nn.Sequential(
            ME.MinkowskiConvolution(self.num_features * 8, self.num_features * 8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 8),
            ME.MinkowskiReLU(),
        )

        self.dec4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(2 * self.num_features * 8, self.num_features * 4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(self.num_features * 4),
            ME.MinkowskiReLU(),
        )

        # (c-1)*s - 2*p + (k-1)+1 +op = (c-1)*s -2*p+k+op 
        self.dec3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(2 * self.num_features * 4, self.num_features * 2, kernel_size=4, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(2 * self.num_features),
            ME.MinkowskiReLU(),
        )
       
    def forward(self, x):
        # Encode
        x_e1 = self.enc1(x)
        x_e2 = self.enc2(x_e1)
        x_e3 = self.enc3(x_e2)
        x_e4 = self.enc4(x_e3)
        # Bottleneck
        x_d4 = self.bottleneck(x_e4)
        # Decode
        x_d3 = self.dec4(ME.cat(x_d4, x_e4 ))
        x_d2 = self.dec3(ME.cat(x_d3, x_e3))
        
        return x_e1, x_e2, x_e3, x_e4, x_d4, x_d3, x_d2


class DenseBackboneEPND2Col(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            nn.Conv3d(4, self.num_features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(self.num_features, self.num_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features * 2),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(self.num_features * 2, self.num_features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features * 4),
            nn.ReLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(self.num_features * 4, self.num_features * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_features * 8),
            nn.ReLU(),
        )

        # int(( c + 2*p-(k-1)-1) / stride+ 1)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(self.num_features * 8, self.num_features * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_features * 8),
            nn.ReLU(),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 8, self.num_features * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_features * 4),
            nn.ReLU(),
        )

        # (c-1)*s - 2*p + (k-1)+1 +op = (c-1)*s -2*p+k+op 
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 4, self.num_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(2 * self.num_features),
            nn.ReLU(),
        )
       
    def forward(self, x):
        # Encode
        
        x_e1 = self.enc1(x)
        x_e2 = self.enc2(x_e1)
        x_e3 = self.enc3(x_e2)
        x_e4 = self.enc4(x_e3)
        # Bottleneck
        x_d4 = self.bottleneck(x_e4)
        # Decode
        x_d3 = self.dec4(torch.cat([x_d4, x_e4 ], 1))
        x_d2 = self.dec3(torch.cat([x_d3, x_e3], 1))
        
        return x_e1, x_e2, x_e3, x_e4, x_d4, x_d3, x_d2