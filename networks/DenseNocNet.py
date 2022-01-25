import torch
from torch import nn
import torch.nn.functional as F

class DenseNocDec2(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.num_features = conf.num_features

        # (c-1)*s - 2*p + (k-1)+1 +op = (c-1)*s -2*p+k+op 
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 2, self.num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(self.num_features* 2, self.num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.Conv3d(self.num_features, self.num_features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(self.num_features),
            nn.ConvTranspose3d(self.num_features, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x_de2_batch_crops, x_e1_batch_crops):
        x_d1_batch_crops = self.dec2(x_de2_batch_crops)
        x_d0_batch_crops = self.dec1(torch.cat([x_d1_batch_crops, x_e1_batch_crops], 1))
        return x_d0_batch_crops


class DenseNocDec2Bigger(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.num_features = conf.num_features

        # (c-1)*s - 2*p + (k-1)+1 +op = (c-1)*s -2*p+k+op 
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 2, self.num_features *2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features *2 ),
            nn.ReLU(),
            # reencode
            nn.Conv3d(self.num_features *2, self.num_features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features *4 ),
            nn.ReLU(),
            nn.Conv3d(self.num_features* 4, self.num_features* 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features *4),
            nn.ReLU(),

            nn.ConvTranspose3d(4 * self.num_features , self.num_features *2 , kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features*2),
            nn.ReLU(),
            nn.ConvTranspose3d(2* self.num_features , self.num_features  , kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features ),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(self.num_features* 2, self.num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.Conv3d(self.num_features, self.num_features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(self.num_features),
            nn.ConvTranspose3d(self.num_features, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x_de2_batch_crops, x_e1_batch_crops):
        x_d1_batch_crops = self.dec2(x_de2_batch_crops)
        x_d0_batch_crops = self.dec1(torch.cat([x_d1_batch_crops, x_e1_batch_crops], 1))
        return x_d0_batch_crops