from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback
import sys
from PIL import Image
import numpy as np
import pandas as pd
import skimage


def get_class_labels(intKey=False):
    img_type_dict = {
        "NORMAL": 0,
        "DRUSEN": 1,
        "CNV": 2,
        "DME": 3,
    }
    if intKey:
        img_type_dict = {i: l for (l, i) in img_type_dict.items()}
    return img_type_dict


def save_data(output_path, img_stack, target_list,
              new_size, dataset, n_train, img_names):
    # save as np array
    img_stack = np.stack(img_stack, axis=0)
    target_list = np.asarray(target_list)
    target_df = pd.DataFrame(index=img_names)
    target_df[dataset] = target_list

    info_tag = "{}_{}".format(str(new_size), dataset)
    if dataset == 'train':
        img_stack_out_path = os.path.join(output_path,
                                          "imgData_{}_n{}.npy".format(info_tag,
                                                                      n_train))
    else:
        img_stack_out_path = os.path.join(output_path,
                                          "imgData_{}.npy".format(info_tag))
    target_list_out_path = os.path.join(output_path,
                                        "targetData_{}.npy".format(info_tag))
    np.save(img_stack_out_path, img_stack)
    np.save(target_list_out_path, target_list)
    target_df.to_csv(os.path.join(output_path,
                                  "targetData_{}.csv".format(info_tag)))


def preprocess_dir(data_path,
                   output_path,
                   dataset,
                   n_train,
                   new_size,
                   ):
    """
    Preprocess directory of .jpeg images.
    Each image is normalized and resized to desired resolution
    Final data is a numpy stack of image files compatible with
    keras model fit

    :param data_path: base dir of raw data (str).

    Directory must be structured as follows
    -data_path
        -NORMAL
            -img1.jpeg
            -img2.jpeg
            ..
        -DME
            -img1.jpeg
            ..
        -CNV
            -img1.jpeg
            ..
        -DRUSEN
            -img1.jpeg

    :param output_path: location for preprocessed data (str)
    :param dataset: specify dataset for training, testing, or validation (str)
    one of: train, test, val
    :param n_train: This specifies the number of samples used
    if dataset is set to train (int)
    :param new_size: resolution of final img (tuple)
    :return: None
    """
    img_type_dict = get_class_labels()

    print('Preprocessing:', dataset)
    target_data_path = data_path
    disease_dirs = os.listdir(target_data_path)
    disease_dirs = [d for d in disease_dirs if
                   os.path.isdir(os.path.join(target_data_path, d))]
    img_stack, target_list = [], []
    img_names = []
    for img_type in disease_dirs:
        class_lbl = img_type_dict[img_type]
        n_class = int(n_train / len(disease_dirs))
        print('\t', img_type)
        img_files_path = os.path.join(target_data_path, img_type)
        if not (os.path.isdir(img_files_path)):
            continue
        img_files = os.listdir(img_files_path)
        img_files = [f for f in img_files if f.endswith('.jpeg')]
        if dataset == 'train':
            img_files = img_files[0:n_class]
        for img_fname in img_files:
            img_path = os.path.join(img_files_path, img_fname)
            img_arr = np.array(Image.open(img_path))
            img_arr = skimage.transform.resize(img_arr, new_size)
            img_arr = (img_arr - img_arr.min()) / img_arr.max()
            img_stack.append(img_arr)
            target_list.append(class_lbl)
        img_names += [n.split('.')[0] for n in img_files]
    # Save preprocessed data
    save_data(output_path, img_stack, target_list,
              new_size, dataset, n_train, img_names)


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i", dest="data_path", type=str,
                               help="the location dataset")
    module_parser.add_argument("-o", dest="output_path", type=str,
                               help='base dir for outputs')
    module_parser.add_argument("-subdir", dest="subdir", type=str,
                               choices=['test', 'train', 'val', 'all'],
                               help='subdir: trn, test, val, or all ...')
    module_parser.add_argument("-n", dest="n_train", type=int,
                               help='n: number of images for training')
    module_parser.add_argument("-Rx", dest="x_res", type=int,
                               help='x resulution for final img')
    module_parser.add_argument("-Ry", dest="y_res", type=int,
                               help='y resolution of final image')
    module_parser.add_argument("-d", dest="d",
                               type=int,
                               default=0,
                               help='debug')
    return module_parser


def main_driver(data_path, output_path, subdir,
                n_train, x_res, y_res, d):
    """
    preprocess data for training CNN
    :param data_path: base path to data directory (str)
    :param output_path: path to output directory (str)
    :param subdir: input data sub directory. train, test, or val (str)
    :param n_train: number of samples to use for training (int)
    :param x_res: desired image width for preprocessing [may be changed for model] (int)
    :param y_res: desired image height for preprocessing [may be changed for model] (int)
    :param d: debugging mode [limit dataset size and training iterations] (int)
    :return: None
    """
    if d == 1:
        print('debug mode: ON')
        subdir = 'train'
        n_train = 10

    assert (os.path.isdir(data_path))
    new_size = (int(x_res), int(y_res), 3)
    if not (os.path.isdir(output_path)):
        os.makedirs(output_path)
    print(output_path)
    if subdir == 'all':
        for subdir in ['test', 'train', 'val']:
            preprocess_dir(os.path.join(data_path, subdir),
                           output_path, subdir, n_train, new_size)
    else:
        preprocess_dir(os.path.join(data_path, subdir),
                       output_path, subdir, n_train, new_size)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.data_path,
                    args.output_path,
                    args.subdir,
                    args.n_train,
                    args.x_res,
                    args.y_res,
                    args.d)
        print('Done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
