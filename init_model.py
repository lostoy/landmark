from torch import optim
def prepare_model(context):
    train_dataset = context['train_dataset']
    args = context['args']

    from torch import nn
    from models.flat_classify import FlatClassify

    model = FlatClassify(args.basenet_name, [t.n_class for t in train_dataset], False)
    crit = []
    for _ in train_dataset:
        crit.append(nn.CrossEntropyLoss())

    #
    # from tools.config_wrapper import ConfigWrapper
    # class Dummy(nn.Module, ConfigWrapper):
    #     def __init__(self):
    #         attrs = locals()
    #         ConfigWrapper.__init__(self, attrs)
    #         nn.Module.__init__(self, )
    #         self.fc1 = nn.Linear(3, 1000)
    #         self.fc2 = nn.Linear(3, 1000)
    #         self.forward_ptr = [0]
    #     def set_forward_ptr(self, ptr):
    #         self.forward_ptr = ptr
    #
    #     def forward(self, inps):
    #         outs = []
    #         inps = inps[:, :, 0, 0]
    #         if 0 in self.forward_ptr:
    #             outs.append(self.fc1(inps))
    #         if 1 in self.forward_ptr:
    #             outs.append(self.fc2(inps))
    #         return outs
    #
    # model = Dummy()
    # crit = []
    # for _ in train_dataset:
    #     crit.append(nn.CrossEntropyLoss())

    if args.gpu_id is not None:
        model = model.cuda(args.gpu_id[0])
    else:
        model = model.float()

    optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    context['model'] = model
    context['crit'] = crit
    context['optimizer'] = optimizer
    return model, crit, optimizer
