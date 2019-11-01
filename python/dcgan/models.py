import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def swish(x):
    return x * F.sigmoid(x)

