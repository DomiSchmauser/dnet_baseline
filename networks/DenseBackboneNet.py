import torch
from torch import nn
import torch.nn.functional as F

class DenseBackboneEPN(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf.num_features

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            nn.Conv3d(1, self.num_features, kernel_size=4, stride=2, padding=1),
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
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 2, self.num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features * 2, 1, kernel_size=4, stride=2, padding=1),
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
        x_d1 = self.dec2(torch.cat([x_d2, x_e2], 1))
        x_d0 = self.dec1(torch.cat([x_d1, x_e1], 1))
        
        return x_e1, x_e2, x_e3, x_e4, x_d4, x_d3, x_d2, x_d1, x_d0


class DenseBackboneEPND2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_features = conf['num_features']

        self.enc1 = nn.Sequential(
            #nn.ConstantPad3d(padding=(3,3,1,1,1,1),value=1),
            nn.Conv3d(1, self.num_features, kernel_size=4, stride=2, padding=1),
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
        
        return x_e1, x_e2, x_d2


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
        
        return x_e1, x_e2, x_d2