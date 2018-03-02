import importlib

from torch.utils.data import DataLoader

from dataset.img_dataset import ImageDataset


def prepare_dataset(context):
    args = context['args']
    basenet_name = args.basenet_name
    basenet_module = importlib.import_module('models.pretrained.{}'.format(basenet_name))
    pretrained_settings = basenet_module.pretrained_settings

    resize_size = pretrained_settings[basenet_name]['imagenet']['resize_size']
    input_size = pretrained_settings[basenet_name]['imagenet']['input_size'][1]
    mean = pretrained_settings[basenet_name]['imagenet']['mean']
    std = pretrained_settings[basenet_name]['imagenet']['std']

    from torchvision import transforms
    train_dataset = []
    train_loader = []
    test_dataset = []
    test_loader = []

    for info_basedir in args.info_basedir:
        t_train_dataset = ImageDataset(info_basedir=info_basedir, phase='train', split='0', to_read=('img', 'label'),
                                       transformer=dict(img=transforms.Compose([
                                           transforms.ToPILImage(),
                                           transforms.Scale(resize_size),
                                           transforms.RandomSizedCrop(input_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean,
                                                                std=std)
                                       ])),
                                       run_n_sample=args.max_step*args.batch_size, shuffle=True)

        t_test_dataset = ImageDataset(info_basedir=info_basedir, phase='valid', split='0', to_read=('img', 'label'),
                                      transformer=dict(img=transforms.Compose([
                                          transforms.ToPILImage(),
                                          transforms.Scale(resize_size),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean,
                                                               std=std)
                                      ])),
                                      run_n_sample=0, shuffle=False)

        t_train_loader = DataLoader(t_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
                                    pin_memory=False)
        t_test_loader = DataLoader(t_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
                                   pin_memory=False)
        train_dataset.append(t_train_dataset)
        train_loader.append(t_train_loader)

        test_dataset.append(t_test_dataset)
        test_loader.append(t_test_loader)

    # import numpy as np
    # mu = [[np.array([1, 1]), np.array([-1, -1])],
    #       [np.array([3, 3]), np.array([-3, -3])]]
    #
    # std = 0.5
    # N = 10
    # for i in range(2):
    #     t_train_dataset = TensorDataset(torch.cat([torch.from_numpy(np.random.randn(N, 2) * std + mu[i][0]),
    #                                                torch.from_numpy(np.random.randn(N, 2) * std + mu[i][1])], 0).float(),
    #                                     torch.cat([torch.zeros(N).long(),
    #                                                torch.ones(N).long()], 0))
    #     t_test_dataset = TensorDataset(torch.cat([torch.from_numpy(np.random.randn(N, 2) * std + mu[i][0]),
    #                                               torch.from_numpy(np.random.randn(N, 2) * std + mu[i][1])], 0).float(),
    #                                    torch.cat([torch.zeros(N).long(),
    #                                               torch.ones(N).long()], 0))
    #
    #     t_train_loader = DataLoader(t_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
    #                                 pin_memory=False)
    #     t_test_loader = DataLoader(t_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
    #                                pin_memory=False)
    #     train_dataset.append(t_train_dataset)
    #     train_loader.append(t_train_loader)
    #
    #     test_dataset.append(t_test_dataset)
    #     test_loader.append(t_test_loader)

    context['train_dataset'] = train_dataset
    context['train_loader'] = train_loader

    context['test_dataset'] = test_dataset
    context['test_loader'] = test_loader
    return (train_dataset, train_loader), (test_dataset, test_loader)