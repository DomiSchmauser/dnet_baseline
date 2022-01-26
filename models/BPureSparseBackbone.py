import torch
from torch import nn
import MinkowskiEngine as ME

class BPureSparseBackboneGeo(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net

    def forward(self, bdscan):

        return self.net(bdscan.btsdf_geo_sparse, bdscan.breg_sparse)
    
    def training_step(self, bdscan):
        return self.forward(bdscan)
    
    def validation_step(self, bdscan):
        return self.forward(bdscan)

    def infer_step(self, bdscan):
        return self.forward(bdscan)

class BPureSparseBackboneCol(nn.Module):
    def __init__(self, conf, net):
        super().__init__()
        self.net = net

    def forward(self, bdscan):
        return self.net(bdscan.btsdf_geo_sparse, bdscan.btsdf_col_sparse, bdscan.breg_sparse) #, bdscan.btsdf_surf_keys)
    
    def training_step(self, bdscan):
        return self.forward(bdscan)
    
    def validation_step(self, bdscan):
        return self.forward(bdscan)
    
    def infer_step(self, bdscan):
        return self.forward(bdscan)