import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()

# Base Folder
CONF.PATH.BASE = os.path.abspath(os.path.dirname(__file__)) #Base dnet path

# Data
CONF.PATH.FRONTDATA = os.path.join('/home/dominik/Schreibtisch/Graph3DMOT/Detection', "front_dataset")
CONF.PATH.FUTURE3D = os.path.join('/home/dominik/Schreibtisch/Graph3DMOT/BlenderProc/resources/front_3D', "3D-FUTURE-model")

# Output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "output")



