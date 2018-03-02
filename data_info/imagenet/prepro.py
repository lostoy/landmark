import json
import os

train_basedir = '../../data/imagenet/train'
val_basedir = '../../data/imagenet/valid'

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
                infos.append(dict(img_name=os.path.join(split, name, os.path.splitext(img_name)[0]), label=w2i[name]))

        with open('./info/{}_img_0.json'.format(split), 'w') as f:
            json.dump(infos, f)


def mv_valid_files(in_dir):
    import tqdm
    with open('/data/yingwei/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', 'r') as f:
        labels = f.readlines()
    with open('./synsets.txt', 'r') as f:
        synsets = f.readlines()
    synsets = [s.strip() for s in synsets]

    labels = [synsets[int(l) - 1] for l in labels]

    for i in tqdm.tqdm(range(len(labels))):
        out_dir = os.path.join(in_dir, labels[i])
        try:
            os.makedirs(out_dir)
        except:
            pass
        img_name = 'ILSVRC2012_val_{:08d}.JPEG'.format(i + 1)
        os.rename(os.path.join(in_dir, img_name), os.path.join(out_dir, img_name))

if __name__ == '__main__':
    synset_names = get_vocab()
    get_infos(synset_names)
    # mv_valid_files('/data/yingwei/dataset/imagenet/ILSVRC2012_img_val')
