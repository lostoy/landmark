import argparse
import random
import string


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_worker', default=8, type=int,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--run_id', default='', type=str, metavar='run_id')
    parser.add_argument('--dump_dir', default='./logs/train_flat', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--step', default=-1, type=int)

    parser.add_argument('--max_step', default=-1, type=int)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--lr_decay_mode', default='step', type=str)
    parser.add_argument('--lr_decay_step', nargs='+', type=int)

    parser.add_argument('--lr_decay_gamma', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--grad_clip', default=100, type=float)

    parser.add_argument('--acc_grad', default=0, type=int)

    parser.add_argument('--save_every_step', default=400, type=float)
    parser.add_argument('--update_every_step', default=1, type=int)
    parser.add_argument('--log_every_step', default=1, type=int)

    parser.add_argument('--evaluate', default=0, type=int)

    parser.add_argument('--info_basedir', nargs='+', type=str)
    parser.add_argument('--split', default='0', type=str)
    parser.add_argument('--basenet_name', type=str)

    args = parser.parse_args()
    if args.run_id == '':
        args.run_id = id_generator()
        print('run_id: {}'.format(args.run_id))

    args.gpu_id = args.gpu_id.split(',')
    args.gpu_id = [int(r) for r in args.gpu_id]
    if -1 in args.gpu_id:
        args.gpu_id = None
    return args
