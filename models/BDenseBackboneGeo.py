import torch
from torch import nn


class BDenseBackboneGeo(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net

    def forward(self, bdscan):
        return self.net(bdscan) #self.net(bdscan.btsdf_geo)
    
    def training_step(self, bdscan):
        return self.forward(bdscan)
    
    def validation_step(self, bdscan):
        return self.forward(bdscan)
    
    def infer_step(self, bdscan):
        return self.forward(bdscan)