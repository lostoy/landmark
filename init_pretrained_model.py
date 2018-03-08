from torch import optim
from models.pretrained import *

def prepare_model(context):
    train_dataset = context['train_dataset']
    args = context['args']

    from torch import nn

    #
    from tools.config_wrapper import ConfigWrapper
    class Dummy(nn.Module, ConfigWrapper):
        def __init__(self, basenet_name, pretrain=True):
            attrs = locals()
            ConfigWrapper.__init__(self, attrs)
            nn.Module.__init__(self, )
            self.net = globals()[basenet_name](pretrained='imagenet' if pretrain else None)
            self.forward_ptr = [0]

        def set_forward_ptr(self, ptr):
            self.forward_ptr = ptr

        def forward(self, inps):
            outs = []
            outs.append(self.net.last_linear(self.net.pool(self.net(inps))))
            return outs

    model = Dummy(args.basenet_name, True)
    crit = []
    for _ in train_dataset:
        crit.append(nn.CrossEntropyLoss())

    if args.gpu_id is not None:
        model = model.cuda(args.gpu_id[0])
    else:
        model = model.float()

    optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    context['model'] = model
    context['crit'] = crit
    context['optimizer'] = optimizer
    return model, crit, optimizer
