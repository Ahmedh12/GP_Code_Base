import importlib
import math
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils as vutils

# Settings
# format short g, %precision=5
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
# number of multiprocessing threads
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)
# NumExpr max threads
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)
# OpenMP max threads (PyTorch and SciPy)
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)



def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def model_info(model, verbose=False):
    import thop

    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters()
              if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' %
              ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu',
               'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(
                      p.shape), p.mean(), p.std()))

    # try:  # FLOPS
    device = next(model.parameters()).device  # get model device
    flops = thop.profile(deepcopy(model.eval()),
                         inputs=(torch.zeros(1, 3, 640, 352).to(device), ),
                         verbose=False)[0] / 1E9 * 2
    fs = ', %.1f GFLOPS' % (flops)  # 640x352 FLOPS
    # except:
    #     fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' %
          (len(list(model.parameters())), n_p, n_g, fs))


# from tyf
def import_fun(fun_dir, module):
    fun = module.split('.')
    m = importlib.import_module(fun_dir + '.' + fun[0])
    return getattr(m, fun[1])


def get_gpu_mem():
    mem = '%.3gG' % (torch.cuda.memory_reserved() /
                     1E9 if torch.cuda.is_available() else 0)  # (GB)
    return mem


def rgb2gray(im):
    assert isinstance(im, torch.Tensor)
    assert im.ndim == 4
    im_gray = im[:,
                 0, :, :] * 0.299 + im[:,
                                       1, :, :] * 0.587 + im[:,
                                                             2, :, :] * 0.114
    return im_gray.unsqueeze(1)


def save_tensor_to_image(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.detach().cpu()
    vutils.save_image(input_tensor, filename)
