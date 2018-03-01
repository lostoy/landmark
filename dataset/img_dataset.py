import os
import torch

from torch.utils.data import Dataset
from tools.config_wrapper import ConfigWrapper
from dataset.utils import img_loader
import json
import numpy as np
import math
import zipfile
import cv2
import collections
LOADER_DICT = dict(img=img_loader,)

class ImageDataset(Dataset, ConfigWrapper):
    def __init__(self, info_basedir, phase='', split='', to_read=(), transformer=None, run_n_sample=0, shuffle=True):
        attrs = locals()
        ConfigWrapper.__init__(self, attrs)
        with open(os.path.join(info_basedir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

        self.n_class = self.dataset_info['n_class']

        infos = dict()
        t_n_sample = None
        for mod_name in self.dataset_info['modality'].keys():
            mod = self.dataset_info['modality'][mod_name]
            if os.path.exists(os.path.join(info_basedir, '{}_{}_{}.json.zip'.format(phase, mod_name, split))):
                with zipfile.ZipFile(os.path.join(info_basedir, '{}_{}_{}.json.zip'.format(phase, mod_name, split)), 'r') as f:

                    infos[mod_name] = json.loads(f.read('{}_{}_{}.json'.format(phase, mod_name, split)).decode("utf-8"))
                    if t_n_sample is None:
                        t_n_sample = len(infos[mod_name])
                    else:
                        if len(infos[mod_name]) != t_n_sample:
                            RuntimeError('sample number wrong for modality {}'.format(mod_name))
            else:
                print('warning, missing info for modal:{}'.format(mod_name))
            # register mode loader
            self.__setattr__('get_{}'.format(mod_name), lambda item, context: self._get_mode(mod_name, item, context))
        self.infos = infos
        # sample iterator
        if run_n_sample == 0:
            run_n_sample = t_n_sample
        self.run_n_sample = run_n_sample
        self.n_sample = t_n_sample

        n_epoch = int(math.ceil(self.run_n_sample * 1.0 / self.n_sample))

        n_sample_ind_iter = []
        for _ in range(n_epoch):
            if shuffle:
                iter_epoch = np.random.permutation(self.n_sample).tolist()
            else:
                iter_epoch = list(range(self.n_sample))
            n_sample_ind_iter = n_sample_ind_iter + iter_epoch
        n_sample_ind_iter = n_sample_ind_iter[:self.run_n_sample]
        self.n_sample_ind_iter = n_sample_ind_iter

    def _get_mode(self, mode, item, context):
        item = self.n_sample_ind_iter[item]
        img_name = self.infos.values()[0][item]['id']

        mode_path = os.path.join(self.dataset_info['modality'][mode]['mode_basedir'],
                                 img_name + self.dataset_info['modality'][mode]['mode_ext'])
        context[mode] = LOADER_DICT[self.dataset_info['modality'][mode]['mode_format']](mode_path,
                                                                                        self.dataset_info['modality'][
                                                                                            mode]['config'])
        if mode in self.transformer and self.transformer[mode]:
            context[mode] = self.transformer[mode](context[mode])

        return context[mode]

    def get_img_name(self, item, context):
        item = self.n_sample_ind_iter[item]
        img_name = self.infos.values()[0][item]['id']
        context['id'] = img_name
        return img_name

    def get_label(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.infos.values()[0][item]
        label = info['label']
        if isinstance(label, list):
            label = ' '.join([str(l) for l in label])
        context['label'] = label
        return label

    def get_negative_imgs(self, item, context):
        pass

    def __getitem__(self, item):
        context = dict()
        output = []
        for key in self.to_read:
            method = getattr(self, 'get_{}'.format(key))
            val = method(item, context)
            output.append(val)
        return output

    def __len__(self):
        return len(self.n_sample_ind_iter)



if __name__ == '__main__':
    #test_3d()
    #test_jpg()
    pass
