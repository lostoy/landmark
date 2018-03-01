import subprocess, os, sys, datetime
from saver import make_log_dirs, Saver
import json
from tensorboardX import SummaryWriter
def prepare_logger(context):
    args = context['args']
    model = context['model']
    train_dataset = context['train_dataset']

    model_dir, train_dir, log_dir = make_log_dirs(args.dump_dir, args.run_id)
    writer = SummaryWriter(log_dir)

    saver = Saver(model_dir=model_dir, max_to_keep=5)
    config_obj = dict(dataset_config=[tt.config if hasattr(tt, 'config') else None for tt in train_dataset],
                      model_config=model.config if hasattr(model, 'config') else None, train_config=vars(args))
    if not os.path.exists(os.path.join(log_dir, 'config.json')):
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config_obj, f, indent=2)

    # Unbuffer output
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

    tee = subprocess.Popen(["tee", os.path.join(train_dir, datetime.datetime.now().strftime('output_%H_%M_%d_%m_%Y.log'))]
                           , stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    print(' '.join(sys.argv))

    context['writer'] = writer
    context['saver'] = saver
    return writer, saver