from __future__ import absolute_import, division, print_function

from options import Options
from trainer import Trainer
import os, shutil, sys, traceback
import torch
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

options = Options()
opts = options.parse()

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

if __name__ == "__main__":

    # Remove old files
    if os.path.exists(CONF.PATH.OUTPUT):
        print('Removing old outputs ...')
        shutil.rmtree(CONF.PATH.OUTPUT)
        os.mkdir(CONF.PATH.OUTPUT)

    # Start multiprocessing
    #torch.multiprocessing.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(opts)
    trainer.train()
