import copy
import csv
import numpy as np
from collections import OrderedDict

def _get_train_stats():
    with open('data/train.csv', 'r') as f:
        reader = list(csv.DictReader(f))
        info_by_ldmk = dict()
        n_img = 0
        for entry in reader:
            ldmk_id = entry['landmark_id']
            if ldmk_id not in info_by_ldmk:
                info_by_ldmk[ldmk_id] = []
            info_by_ldmk[ldmk_id].append(entry)
            n_img += 1
        info_count_sorted = sorted(info_by_ldmk.items(), key=lambda x: len(x[1]), reverse=True)
        print('# landmarks: {}, # total training images: {}'.format(len(info_count_sorted), n_img))

        print('top 10 landmarks | counts')
        for i in range(10):
            print('landmark: {} | #: {}'.format(info_count_sorted[i][0], len(info_count_sorted[i][1])))
    info_by_ldmk = OrderedDict(info_count_sorted)
    return info_by_ldmk

def _get_img_by_ldmk(info_by_ldmk, ldmk_id, img_inds):
    import urllib
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    from PIL import Image
    img_inds = [ind for ind in img_inds if ind < len(info_by_ldmk[ldmk_id])]
    infos = info_by_ldmk[ldmk_id][img_inds]
    res = copy.deepcopy(infos)

    for info in res:
        try:
            file = StringIO(urllib.urlopen(info['url']).read())
            img = Image.open(file)
        except:
            img = None

        res['img'] = img

    return res