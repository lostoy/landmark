import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from tools.config_wrapper import ConfigWrapper
from models.nasnet import nasnetalarge_feature

class FlatClassify(nn.Module, ConfigWrapper):
    def __init__(self, n_class):
        attrs = locals()
        ConfigWrapper.__init__(self, attrs)
        nn.Module.__init__(self,)

        self.basenet = nasnetalarge_feature('imagenet')

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()

        self.fcs = [nn.Linear(4032, n) for n in n_class]

        self.forward_ptr = [0]

    def forward_pool(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x
    def set_forward_ptr(self, ptr):
        self.forward_ptr = ptr

    def forward(self, inp):
        features = self.forward(inp)
        outs = []
        for i in range(len(self.n_class)):
            if i in self.forward_ptr:
                fc = self.fcs[i](features)
                outs.append(fc)
        return outs