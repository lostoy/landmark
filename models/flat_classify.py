import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from tools.config_wrapper import ConfigWrapper
from models.pretrained.bninception import bninception_feature

class FlatClassify(nn.Module, ConfigWrapper):
    def __init__(self, n_class):
        attrs = locals()
        ConfigWrapper.__init__(self, attrs)
        nn.Module.__init__(self,)

        self.basenet = bninception_feature('imagenet')

        self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)

        for i, n in enumerate(n_class):
            self.__setattr__('fcs_{}'.format(i), nn.Linear(1536, n))

        self.forward_ptr = [0]

    def forward_pool(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        return x
    def set_forward_ptr(self, ptr):
        self.forward_ptr = ptr

    def forward(self, inp):
        features = self.basenet(inp)
        features = self.forward_pool(features)

        outs = []
        for i in range(len(self.n_class)):
            if i in self.forward_ptr:
                fc = self.__getattr__('fcs_{}'.format(i))(features)
                outs.append(fc)
        return outs