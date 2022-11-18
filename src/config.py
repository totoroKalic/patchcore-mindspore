"""config"""
from yacs.config import CfgNode as CN

_C = CN()

_C.mean_dft = [0.485, 0.456, 0.406]
_C.std_dft = [0.229, 0.224, 0.255]
_C.mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255]
_C.std = [1/0.229, 1/0.224, 1/0.255]
