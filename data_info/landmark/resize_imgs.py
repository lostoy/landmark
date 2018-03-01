# -*- coding: utf-8 -*-

# !/usr/bin/python

# Note: requires the tqdm package (pip install tqdm)

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm as tqdm

def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    return 0

def resize_img(img_name):
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.exists(os.path.join(in_dir, img_name)):
        download_image(img_name)
    try:
        img = Image.open(os.path.join(in_dir, img_name))
        img = img.resize((354, 354), Image.ANTIALIAS)
        img.save(os.path.join(out_dir, img_name))
        os.remove(os.path.join(in_dir, img_name))
    except:
        print('cannot do {}'.format(img_name))
        img = Image.new('RGB', (354, 354))
        img.save(os.path.join(out_dir, img_name))
        return 1
    return 0

def loader():
    if len(sys.argv) != 4:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)

    out_dir = sys.argv[2]
    data_file = sys.argv[3]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)

    exist_img_names = os.listdir(out_dir)

    img_names = []
    for en in key_url_list:
        if en[0]+'.jpg' not in exist_img_names:
            img_names.append(en)

    pool = multiprocessing.Pool(processes=20)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(resize_img, img_names), total=len(img_names)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    loader()
