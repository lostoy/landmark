import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from tools.config_wrapper import ConfigWrapper
from models.pretrained.inceptionresnetv2 import inceptionresnetv2

class FlatClassify(nn.Module, ConfigWrapper):
    def __init__(self, n_class):
        attrs = locals()
        ConfigWrapper.__init__(self, attrs)
        nn.Module.__init__(self,)

        self.basenet = inceptionresnetv2('imagenet')

        self.relu = nn.ReLU()
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)

        self.fcs = [nn.Linear(1536, n) for n in n_class]

        self.forward_ptr = [0]

    def forward_pool(self, features):
        x = self.avgpool_1a(features)
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
                fc = self.fcs[i](features)
                outs.append(fc)
        return outs