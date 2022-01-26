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
        # TODO STRIDE USED AT KERNEL_SIZE?
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
        
        return x_e2

class PureSparseBackboneCol_Res1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(4, self.num_features, kernel_size=5, stride=2, dimension=3),
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
        
        return x_e2




class SparseUNetRPN(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.nf = conf.num_features

        ch = [self.nf//2, self.nf, 2*self.nf, 4*self.nf, 2**3*self.nf, 2**4*self.nf]

        self.eb0 = nn.Sequential(
            ME.MinkowskiConvolution(4, ch[0], kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiReLU(True),
            )
        self.ec0 = ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=2, stride=2, dimension=3)
            
        self.eb1 = nn.Sequential(
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiReLU(True),
            BasicBlock(ch[1], ch[1], dimension=3),
            )
        self.ec1 = ME.MinkowskiConvolution(ch[1] , ch[2] , kernel_size=2, stride=2, dimension=3)
            
        self.eb2 = nn.Sequential(
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiReLU(True),
            BasicBlock(ch[2], ch[2], dimension=3),
            )
        self.ec2 = ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=2, stride=2, dimension=3)
     
        self.eb3 = nn.Sequential(
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiReLU(True),
            BasicBlock(ch[3], ch[3], dimension=3),
            )
        self.ec3 = ME.MinkowskiConvolution(ch[3] , ch[4] , kernel_size=2, stride=2, dimension=3)

        self.eb4 = nn.Sequential(
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiReLU(True),
            BasicBlock(ch[4], ch[4], dimension=3),
            )
        self.ec4 = ME.MinkowskiConvolution(ch[4] , ch[5] , kernel_size=2, stride=2, dimension=3)

        # segmentation only at first
        self.bottle_neck = nn.Sequential(
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiReLU(True),
            BasicBlock(ch[5], ch[5], dimension=3),
            ) 
    

        self.db4 = nn.Sequential(
            BasicBlock(2*ch[5], 2*ch[5], dimension=3),
            ME.MinkowskiConvolution(
            2*ch[5], 2*ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(2*ch[5]),
            ME.MinkowskiReLU(True),
            )
        
        self.du4 =  ME.MinkowskiConvolutionTranspose(2*ch[5], ch[4], kernel_size=2, stride=2, dimension=3)
        #ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)
        
        self.db3 = nn.Sequential(
            BasicBlock(2*ch[4], 2*ch[4], dimension=3),
            ME.MinkowskiConvolution(
            2*ch[4], 2*ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(2*ch[4]),
            ME.MinkowskiReLU(True),
            )
        
        self.du3 = ME.MinkowskiConvolutionTranspose(2*ch[4], ch[3], kernel_size=2, stride=2, dimension=3)
        #ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)
        
        self.db2 = nn.Sequential(
            BasicBlock(2*ch[3], 2*ch[3], dimension=3),
            ME.MinkowskiConvolution(
            2*ch[3], 2*ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(2*ch[3]),
            ME.MinkowskiReLU(True),
            )

        
        self.du2 = ME.MinkowskiConvolutionTranspose(2*ch[3], ch[2], kernel_size=2, stride=2, dimension=3)

        self.db1 = nn.Sequential(
            BasicBlock(2*ch[2], 2*ch[2], dimension=3),
            ME.MinkowskiConvolution(
            2*ch[2], 2*ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(2*ch[2]),
            ME.MinkowskiReLU(True),
            )

        self.rematching = ME.MinkowskiConvolution(
            2*ch[2], 2*ch[2], kernel_size=3, stride=1, dimension=3)

        
        self.du1 = ME.MinkowskiConvolutionTranspose(2*ch[2], ch[1], kernel_size=2, stride=2, dimension=3)

        self.db0 = nn.Sequential(
            BasicBlock(2*ch[1], 2*ch[1], dimension=3),
            ME.MinkowskiConvolution(
            2*ch[1], 2*ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(2*ch[1]),
            ME.MinkowskiReLU(True),
            )
        
        self.du0 = ME.MinkowskiConvolutionTranspose(2*ch[1], ch[0], kernel_size=2, stride=2, dimension=3)
        
       
       
    def forward(self, x_geo, x_col, c_out, tsdf_surf_key_lvl):
        cm = x_geo.coords_man
        #with Timer('sparse encode 1'):
        e0 = ME.cat(x_geo, x_col)
        e0_1 = self.eb0(e0)
        e1 = self.ec0(e0_1, tsdf_surf_key_lvl[1])
        #with Timer('sparse encode 2-'):
        e1_1 = self.eb1(e1)
        e2 = self.ec1(e1_1, tsdf_surf_key_lvl[2])

        e2_1 = self.eb2(e2)
        e3 = self.ec2(e2_1, tsdf_surf_key_lvl[3])

        e3_1 = self.eb3(e3)
        e4 = self.ec3(e3_1, tsdf_surf_key_lvl[4])

        e4_1 = self.eb4(e4)
        e5 = self.ec4(e4_1, tsdf_surf_key_lvl[5])
    
        bneck = self.bottle_neck(e5)
        d4 = bneck
        #### d4
        #with Timer('sparse decode'):
        de_4 = ME.cat(d4, e5)
        de4_f = self.db4(de_4)
        d3 = self.du4(de4_f)

        de_3 = ME.cat(d3, e4)
        de3_f = self.db3(de_3)
        d2 = self.du3(de3_f)

        de_2 = ME.cat(d2, e3)
        de2_f = self.db2(de_2)
        d1 = self.du2(de2_f)

        de_1 = ME.cat(d1, e2)

        de1_rematched = self.rematching(de_1, c_out.C)
        de1_rematched.set_tensor_stride([4,4,4])

        de1_f = self.db1(de1_rematched)
        
        


        """
        d0 = self.du1(de1_f)

        de_0 = ME.cat(d0, e1)
        de0_f = self.db0(de0)
        """
        return de1_f





class PureSparseBackboneCol_Res1(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.num_features = conf['num_features']

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            ME.MinkowskiConvolution(4, self.num_features, kernel_size=5, stride=2, dimension=3),
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
        x = ME.cat(x_geo,x_col)
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