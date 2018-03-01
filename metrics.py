import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def poly_adjust_learning_rate(optimizer, lr0, step, n_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (1.0 - step*1.0/n_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def step_adjust_learning_rate(optimizer, lr0, step, step_size, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if len(step_size) == 0:
        lr = lr0 * (gamma ** (step // step_size))
    else:
        lr = lr0 * gamma ** (sum([step > i for i in step_size]))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_for_nan(ps):
    for p in ps:
        if p is not None:
            if not np.isfinite(np.sum(p.data.cpu().numpy())):
                return True
    return False
