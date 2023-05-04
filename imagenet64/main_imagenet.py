"""
This is for and only for ImageNet 64*64.
dataset     : https://image-net.org/download-images.php
description : https://patrykchrabaszcz.github.io/Imagenet32/
download URL: https://image-net.org/data/downsample/Imagenet64_train_part1.zip
              https://image-net.org/data/downsample/Imagenet64_train_part2.zip
              https://image-net.org/data/downsample/Imagenet64_val.zip
"""
import os
import pickle
import numpy as np
import torch
import torchvision.utils as tvu

def load_data_batch(data_folder, idx, img_size=64):
    data_file = os.path.join(data_folder, f"train_data_batch_{idx}")
    print(f"load file: {data_file}")
    with open(data_file, 'rb') as fo:
        d = pickle.load(fo)
    x = d['data']           # 0 ~ 255
    y = d['labels']         # 1 ~ 1000
    x = x/np.float32(255)   # 0.0 ~ 1.0
    y = [i-1 for i in y]    # 0 ~ 999
    # mean_image = d['mean']
    # mean_image = mean_image/np.float32(255)
    # x -= mean_image

    ss = img_size * img_size
    x = np.dstack((x[:, :ss], x[:, ss:2*ss], x[:, 2*ss:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    return x, y

def get_id_map(data_id_file, gen_id_file):
    """
    This is a bit complicated. the class ID in data (ImageNet64) is different with
    the class ID in generation model. For example, the class ID 153:
        In ImageNet64 data, it is : n02128757: snow_leopard
        In generation model, it is: n02085782: Japanese spaniel
    But luckily, both ID mappings share the same label. So here we map the two ID system
    with the label.
    :param data_id_file:
    :param gen_id_file:
    :return:
    """
    print(f"get_id_map()...")
    print(f"  data_id_file: {data_id_file}")
    print(f"  gen_id_file : {gen_id_file}")
    with open(data_id_file, 'r') as fptr:
        lines = fptr.readlines()
    id2label = {}  # Imagenet64 data ID to class label
    label2id = {}  # generation model label to ID
    for line in lines:  # line sample: n02128757 153 snow_leopard
        line = line.strip()
        if line == '' or line.startswith('#'): continue
        label, did, _ = line.split(' ', 2)
        did = int(did) - 1  # ID start from 0
        if did in id2label:
            raise ValueError(f"Duplicate ID {did} in {data_id_file}.")
        id2label[did] = label
    # for
    with open(gen_id_file, 'r') as fptr:
        lines = fptr.readlines()
    id_counter = 0
    for line in lines:  # line sample: n02085782: Japanese spaniel
        line = line.strip()
        if line == '' or line.startswith('#'): continue
        label, _ = line.split(':', 1)
        if label in label2id:
            raise ValueError(f"Duplicate label {label} in {gen_id_file}")
        label2id[label] = id_counter
        id_counter += 1
    # for
    len1 = len(id2label)
    len2 = len(label2id)
    print(f"  id2label: {len1}")
    print(f"  label2id: {len2}")
    if len(id2label) != len(label2id):
        raise ValueError(f"Two map have different size: id2label {len1}, label2id {len2}")
    id2id = {}
    for id1, label in id2label.items():
        if label not in label2id:
            raise ValueError(f"Label {label} not found in label2id map")
        id2 = label2id[label]
        id2id[id1] = id2
    # for
    print(f"get_id_map()...done")
    return id2id

def main():
    data_folder = './symlink/imagenet64'
    out_folder  = './symlink/imagenet64'
    data_id_file = './imagenet64/class_id_label_in_data.txt'
    gen_id_file  = './imagenet64/class_id_label_in_generation.txt'
    id2id = get_id_map(data_id_file, gen_id_file)
    img_size = 64
    print(f"data_folder: {data_folder}")
    print(f"out_folder : {out_folder}")
    print(f"img_size   : {img_size}")
    cls_cnt_arr = [0] * 1000
    total_cnt = 0
    for idx in range(1, 11):
        x_arr, y_arr = load_data_batch(data_folder, idx, img_size=img_size)
        print(f"idx: {idx:02}/10. batch size : {len(x_arr)}")
        for x, y in zip(x_arr, y_arr):
            y = id2id[y]
            d_path = os.path.join(out_folder, f"cls{y:03d}")
            if cls_cnt_arr[y] == 0 and not os.path.exists(d_path):
                os.makedirs(d_path)
            f_path = os.path.join(d_path, f"{cls_cnt_arr[y]:04d}.png")
            cls_cnt_arr[y] += 1
            x = torch.Tensor(x)
            tvu.save_image(x, f_path)
            total_cnt += 1
            if total_cnt % 1000 == 0: print(f"total_cnt: {total_cnt}")
        # for
    # for idx
    print(f"total_cnt: {total_cnt}")
# main()

if __name__ == '__main__':
    main()
