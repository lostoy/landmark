import json
import csv
import random
import os
import numpy as np
import tqdm as tqdm

np.random.seed(123)

data_basedir = './data/train/'
from tools.vis_web import _get_train_stats as get_train_stats

def get_train_test_split(p=0.9):
    info_by_ldmk = get_train_stats()
    train_infos = []
    test_infos = []

    print('splitting with p: {}'.format(p))

    print('original image #: {}'.format(sum(len(en) for en in info_by_ldmk.values())))

    # filter out images don't exist
    for ldmk_id, entries in tqdm.tqdm(list(info_by_ldmk.items())):
        t_entries = []
        for en in entries:
            if os.path.exists(os.path.join(data_basedir, en['id']+'.jpg')):
                t_entries.append(en)
        if len(t_entries) == 0:
            del info_by_ldmk[ldmk_id]
        else:
            info_by_ldmk[ldmk_id] = t_entries

    print('exist image #: {}'.format(sum(len(en) for en in info_by_ldmk.values())))
    # get vocab
    vocab = list(info_by_ldmk.keys())
    word2ind = {w:i for i, w in enumerate(vocab)}

    print('valid vocab size: {}'.format(len(vocab)))

    with open('./data_info/landmark/info/vocab.json', 'w') as f:
        json.dump(vocab, f, indent=2)

    # encode label
    for ldmk_id, entries in info_by_ldmk.items():
        for en in entries:
            en['label'] = word2ind[ldmk_id]

    # split
    for ldmk_id, entries in tqdm.tqdm(list(info_by_ldmk.items())):
        t_n_sample = len(entries)
        t_train_inds = np.random.choice(t_n_sample, int(t_n_sample*p), False)
        t_test_inds = np.array(list(set(range(t_n_sample)) - set(t_train_inds)))

        if len(t_test_inds) != 0:
            train_infos = train_infos + list(np.array(entries)[t_train_inds])
        if len(t_test_inds) != 0:
            test_infos = test_infos + list(np.array(entries)[t_test_inds])

    print('train #: {}'.format(len(train_infos)))
    print('test #: {}'.format(len(test_infos)))

    with open('./data_info/landmark/info/train_img_0.json', 'w') as f:
        json.dump(train_infos, f, indent=2)

    with open('./data_info/landmark/info/valid_img_0.json', 'w') as f:
        json.dump(test_infos, f, indent=2)

    return train_infos, test_infos

if __name__ == '__main__':
    get_train_test_split()