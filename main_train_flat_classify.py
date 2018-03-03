import os
import sys
import time

import torch
import torch.multiprocessing
import torch.nn.utils
from torch.autograd.variable import Variable

from cmdline import parse_args
from init_dataset import prepare_dataset
from init_logger import prepare_logger
from init_model import prepare_model
from metrics import step_adjust_learning_rate, poly_adjust_learning_rate, AverageMeter, accuracy, check_for_nan


def prepare_stats(context):
    context['step'] = 0
    context['best_metric'] = None

    context['stats_train'] = dict(batch_time=AverageMeter(),
                                  data_time=AverageMeter(),
                                  )

    context['stats_val'] = dict(batch_time=AverageMeter(),
                                data_time=AverageMeter(),
                                )
    context['timer'] = Timer()

    return context['step'], context['best_metric'], context['stats_train'], context['stats_val'], context['timer']

def main():
    # global step, best_prec1, model, crit, optimizer, saver, writer
    # global train_dataset, test_dataset, train_loader, test_loader
    args = parse_args()
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id[0])

    context = dict(args=args)
    print('prepare dataset...')
    (train_dataset, train_loader), (test_dataset, test_loader) = prepare_dataset(context)

    print('prepare model...')
    model, crit, optimizer = prepare_model(context)

    print('prepare logger...')
    writer, saver = prepare_logger(context)

    step, best_prec1, stats_train, stats_test, timer = prepare_stats(context)
    print('start training...')
    train(context)

def train(context):
    loader = context['train_loader']
    model = context['model']
    timer = context['timer']
    data_time = context['stats_train']['data_time']
    batch_time = context['stats_train']['batch_time']

    # train pointers
    iter_ptr = [iter(t_loader) for t_loader in loader]
    dataset_ptr = 0
    timer.tic()
    while True:
        timer.tic()
        # advance dataset pointer, the datasets are visited in a round-robin style
        inputs = None
        n_trial = 0
        while not inputs:
            inputs = next(iter_ptr[dataset_ptr], None)
            dataset_ptr = (dataset_ptr + 1) % len(loader)
            n_trial += 1
            if n_trial >= len(iter_ptr):
                break

        if not inputs:
            print('dataloader iteration finished, reinitializing dataloader...')
            iter_ptr = [iter(t_loader) for t_loader in loader]
            dataset_ptr = 0
            while not inputs:
                inputs = next(iter_ptr[dataset_ptr], None)
                dataset_ptr = (dataset_ptr + 1) % len(loader)
                n_trial += 1
                if n_trial >= len(iter_ptr):
                    break

        batch = dict()
        batch['dataset_ptr'] = dataset_ptr

        # load sample
        # @interface
        imgs, labels = inputs
        data_time.update(timer.toc())

        # main forward
        # @interface
        batch.update(dict(imgs=imgs, labels=labels))
        outputs = forward_net(batch, context, True)
        batch.update(outputs)

        # stats
        forward_stats(batch, context)

        # main backward
        backward_net(batch, context)

        batch_time.update(timer.toc())

        if context['step'] % context['args'].log_every_step == 0:
            forward_log(batch, context, training=True, dump_meter=True)

        validate(context, force_validate=False, is_test=False, force_save=False)

        # update training pointers
        context['step'] += 1

        if context['step'] >= context['args'].max_step:
            break

        if check_for_nan(model.parameters()):
            print('nan in parameters')
            sys.exit(-1)
        timer.tic()

def validate(context, force_validate=False, is_test=False, force_save=False):
    loader = context['test_loader']
    model = context['model']
    timer = context['timer']
    data_time = context['stats_val']['data_time']
    batch_time = context['stats_val']['batch_time']
    step = context['step']
    args = context['args']
    saver = context['saver']

    if not(force_validate or step % args.save_every_step == 0 or step >= args.max_step - 1):
        return

    # dataset pointers
    iter_ptr = [iter(t_loader) for t_loader in loader]
    dataset_ptr = 0
    timer.tic()

    val_iter = 0

    while True:
        timer.tic()
        # advance dataset pointer, the datasets are visited in a round-robin style
        inputs = None
        n_trial = 0
        while not inputs:
            inputs = next(iter_ptr[dataset_ptr], None)
            dataset_ptr = (dataset_ptr + 1) % len(loader)
            n_trial += 1
            if n_trial >= len(iter_ptr):
                break
        if not inputs:
            break

        batch = dict()
        batch['dataset_ptr'] = dataset_ptr

        # load sample
        # @interface
        imgs, labels = inputs
        data_time.update(timer.toc())

        # main forward
        # @interface
        batch.update(dict(imgs=imgs, labels=labels))
        outputs = forward_net(batch, context, training=False)
        batch.update(outputs)

        # stats
        forward_stats(batch, context, training=False)

        # main backward
        backward_net(batch, context)

        batch_time.update(timer.toc())
        if val_iter % args.log_every_step == 0:
            forward_log(batch, context, training=False, dump_meter=False)
        val_iter += 1

        if step == 0:
            break
        timer.tic()

    forward_log(None, context, training=False, dump_meter=True)

    # save model
    if force_save or (not force_validate and not is_test):
        is_best, metric = forward_check_metric(context)

        info_dict = {
            'best_metric': context['best_metric'],
            'step': step
        }
        my_save_checkpoint(saver, model=model, info_dict=info_dict, is_best=is_best, step=step)

def forward_net(inputs, context, training=True):
    # forward
    dataset_ptr = inputs['dataset_ptr']
    model = context['model']
    crit = context['crit']
    args = context['args']

    # @interface
    imgs, labels = inputs['imgs'], inputs['labels']

    if args.gpu_id:
        labels = labels.cuda(args.gpu_id[0])

    imgs_var = Variable(imgs)
    labels_var = Variable(labels)

    if training:
        model.train()
    else:
        model.eval()
    model.set_forward_ptr([dataset_ptr])
    if args.gpu_id:
        ys = torch.nn.parallel.data_parallel(model, imgs_var, args.gpu_id)[0]
    else:
        ys = model(imgs_var)[0]
    loss = crit[dataset_ptr](ys, labels_var)

    return dict(loss=loss.data.cpu(), ys=ys.data.cpu(), ys_var=ys, loss_var=loss)

def backward_net(batch, context):
    # @start interface
    loss_var = batch['loss_var']
    optimizer = context['optimizer']
    step = context['step']
    args = context['args']
    model = context['model']

    loss_var.backward()
    # @end interface

    # compute gradient and do SGD step
    if args.lr_decay_mode == 'poly':
        lr = poly_adjust_learning_rate(optimizer=optimizer, lr0=args.lr, step=step, n_step=args.max_step)

    elif args.lr_decay_mode == 'step':
        lr = step_adjust_learning_rate(optimizer=optimizer, lr0=args.lr, step=step, step_size=args.lr_decay_step,
                                       gamma=args.lr_decay_gamma)

    elif args.lr_decay_mode == 'fix':
        lr = args.lr
    else:
        raise ValueError('lr_decay_mode wrong')

    total_norm = 0
    if (not args.acc_grad) or (step % args.update_every_step == 0):
        total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    batch['lr'] = lr
    batch['total_norm'] = total_norm
    return batch['lr'], batch['total_norm']


def forward_stats(batch, context, training=True):
    # @start interface
    dataset_ptr = batch['dataset_ptr']

    stats = context['stats_train' if training else 'stats_val']

    meter_names = ['{}_loss_meter'.format(dataset_ptr),
                   '{}_top1_meter'.format(dataset_ptr),
                   '{}_top5_meter'.format(dataset_ptr),
                   # '{}_top100_meter'.format(dataset_ptr)
                   ]

    meter_funcs = [lambda batch: float(batch['loss'].numpy()),
                   lambda batch: float(accuracy(batch['ys'], batch['labels'], [1])[0][0]),
                   lambda batch: float(accuracy(batch['ys'], batch['labels'], [5])[0][0]),
                   # lambda batch: accuracy(batch['ys'], batch['labels'], [100])[0],
                   ]

    # @end interface

    for meter_name, meter_func in zip(meter_names, meter_funcs):
        if meter_name not in stats or not training:
            stats[meter_name] = AverageMeter()
        stats[meter_name].update(meter_func(batch), context['args'].batch_size)


def forward_log(batch, context, training=True, dump_meter=True):
    import datetime
    args = context['args']
    writer = context['writer']
    step = context['step']
    stats = context['stats_train' if training else 'stats_val']

    if batch is not None:
        lr = batch['lr']
        total_norm = batch['total_norm']

        print('========================{}======================='.format('training' if training else 'validating'))
        print('  time: {time} \t run_id: {run_id}\t'
              'step: [{step}/{max_step}]\n'
              '  lr {lr:.6f}\td_norm {total_norm:.3f}\n'
            .format(
            step=step, max_step=args.max_step, batch_time=stats['batch_time'],
            data_time=stats['data_time'], lr=lr, run_id=args.run_id,
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), total_norm=total_norm))
        for mt_name, mt in stats.items():
            print('  {mt_name} {mt.val:.4f} ({mt.avg:.4f})\t'.format(mt_name=mt_name, mt=mt))  # , end='', flush=True)
        print('\n')
        print('================================================')

    if dump_meter:
        for mt_name, mt in stats.items():
            writer.add_scalar('{}/{}'.format('train' if training else 'val', mt_name), mt.avg, step)

def forward_check_metric(context):
    # @start interface
    stats = context['stats_val']
    metric = stats['0_top1_meter'].avg
    is_best = (context['best_metric'] is None or metric > context['best_metric'])
    if is_best:
        context['best_metric'] = metric
    return is_best, metric

class Timer(object):
    def __init__(self):
        self.start_time = time.time()
    def tic(self):
        self.start_time = time.time()
    def toc(self):
        return time.time() - self.start_time


def save_score(fn, vid_paths, ys):
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))

    with open(fn, 'w') as f:
        for vid_path, y in zip(vid_paths, ys):
            f.write('{}'.format(vid_path))
            for y_ in y:
                f.write('\t{}'.format(y_))
            f.write('\n')


def my_save_checkpoint(saver, model, info_dict, is_best, step):
    saver.save(model=model, info_dict=info_dict, is_best=is_best, step=step)

if __name__ == '__main__':
    main()
