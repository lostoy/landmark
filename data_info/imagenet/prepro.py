import os
import json

train_basedir = '../../data/imagenet/train'
val_basedir = '../../data/imagenet/val'

def get_vocab():
    synset_names = sorted(os.listdir(train_basedir))
    with open('./info/vocab.json', 'w') as f:
        json.dump(synset_names, f, indent=2)
    return synset_names
def get_infos(synset_names):
    w2i = {w:i for i, w in enumerate(synset_names)}
    train_infos = []
    val_infos = []
    for split in ['train', 'valid']:
        if split == 'train':
            basedir = train_basedir
            infos = train_infos
        else:
            basedir = val_basedir
            infos = val_infos
        for name in synset_names:
            img_names = sorted(os.listdir(os.path.join(basedir, name)))
            for img_name in img_names:
                infos.append(dict(img_name=os.path.join(name, os.path.splitext(img_name)[0]), label=w2i[name]))

        with open('./info/{}_img_0.json'.format(split), 'w') as f:
            json.dump(infos, f, indent=2)

if __name__ == '__main__':
    synset_names = get_vocab()
    get_infos(synset_names)
