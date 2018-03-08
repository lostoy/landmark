import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from tools.config_wrapper import ConfigWrapper
from models.pretrained import *

class FlatClassify(nn.Module, ConfigWrapper):
    def __init__(self, basenet_name, n_class, pretrain=True):
        attrs = locals()
        ConfigWrapper.__init__(self, attrs)
        nn.Module.__init__(self,)

        self.basenet = globals()[basenet_name](pretrained='imagenet' if pretrain else None)

        for i, n in enumerate(n_class):
            self.__setattr__('fcs_{}'.format(i), nn.Linear(self.basenet.last_linear.in_features, n))

        self.forward_ptr = [0]

    def set_forward_ptr(self, ptr):
        self.forward_ptr = ptr

    def forward(self, inp):
        features = self.basenet(inp)
        if hasattr(self.basenet, 'pool'):
            features = self.basenet.pool(features)

        outs = []
        for i in range(len(self.n_class)):
            if i in self.forward_ptr:
                fc = self.__getattr__('fcs_{}'.format(i))(features)
                outs.append(fc)
        return outs