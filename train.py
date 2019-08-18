"""
@authors: Andrei Mouraviev & Eric TK Chou
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import traceback
import sys
import numpy as np
from numpy.random import seed
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow import set_random_seed
import datetime
sys.path.append('./methods')
from preprocess import preprocess_dir
from trainCNN import train_model


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-iTrn", dest="x_trn_path", type=str,
                               help="training data directory")
    module_parser.add_argument("-iVal", dest="x_val_path",
                               type=str, default=None,
                               help="validation data directory")
    module_parser.add_argument("-n", dest="n_train", type=int, default=1000,
                               help='number of images for training')
    module_parser.add_argument("-d", dest="d", type=int,
                               default=0, help='debug mode')
    return module_parser


def preprocess_data(x_trn_path, x_val_path, n_train):
    """
    preliminary preprocessing to noralize image size and intensity
    preprocess for 224x224 and 299x299 resolutions
    :param x_trn_path: path to training data (str)
    :param x_val_path: path to val data (str or None)
    :param n_train: number of training samples (int)
    :return: dictionary of paths to preprocessed data indexed by resolution (dict)
    """
    assert (os.path.isdir(x_trn_path))
    if x_val_path is not None:
        assert (os.path.isdir(x_val_path))

    # preprocess data for each type of network
    data_path_dict = {}
    for res in [224, 299]:
        new_size = (res, res, 3)
        output_data_path = join(".", "PreprocessedData", str(new_size))
        if not os.path.isdir(output_data_path):
            os.makedirs(output_data_path)
        preprocess_dir(x_trn_path, output_data_path,
                      'train', n_train, new_size)
        if x_val_path is not None:
            preprocess_dir(x_val_path, output_data_path,
                          'val', n_train, new_size)
        data_path_dict[res] = output_data_path
    return data_path_dict

def save_info(output_model_path,
              trn_data_path,
              val_data_path=None):
    """
    save training info
    :param outputModel Path: write path (str)
    :param trn_data_path: path to training data  (str)
    :param val_data_path: path to validation data (str)
    :return: None
    """

    with open(os.path.join(output_model_path, 'dataInfo.txt'), 'w') as fid:
        fid.write("x_train_path: {} \n".format(trn_data_path))
        if val_data_path is not None:
            fid.write("x_val_path: {} \n".format(val_data_path))
        else:
            fid.write("x_val_path: {} \n".format(trn_data_path))


def train_models(models, data_path_dict, n_train, has_val, d):
    """
    Train each model in models
    :param models: list of model names (list)
    :param data_path_dict: dict containing path to data (dict)
    :param n_train: number of training samples to use (int)
    :param has_val: validation data present (bool)
    :param d: debug mode 1=ON, 0=OFF (int/bool)
    :return: None
    """
    now = datetime.datetime.now()
    today = str(now.date())
    for model_name in models:
        if model_name == "VGG16":
            res = 224
        else:
            res = 299
        # load data for each network
        output_data_path = data_path_dict[res]
        print('loading data')
        trn_tag = "{}_{}".format(str((res, res, 3)), 'train')
        trn_data_path = join(output_data_path, "imgData_{}_n{}.npy".format(trn_tag, n_train))
        trn_target_path = join(output_data_path, "targetData_{}.npy".format(trn_tag))
        x_trn = np.load(trn_data_path)
        y_trn = np.load(trn_target_path)

        if has_val:
            val_tag = "{}_{}".format(str((res, res, 3)), 'val')
            val_data_path = join(output_data_path, "imgData_{}.npy".format(val_tag))
            val_target_path = join(output_data_path, "targetData_{}.npy".format(val_tag))
            x_val = np.load(val_data_path)
            y_val = np.load(val_target_path)
        else:
            val_data_path = None
            x_val = None
            y_val = None

        # save each CNN in a subdirectory
        output_model_path = "./modelOutput/metaClf_{}/{}".format(today,
                                                                 model_name)
        if not os.path.isdir(output_model_path):
            os.makedirs(output_model_path)
        train_model(x_trn, y_trn,
                    x_val, y_val,
                    output_model_path,
                    model_name, 'imagenet',
                    aug=0, d=d, note="")
        # save details of training in each CNN directory
        save_info(output_model_path,
                  trn_data_path,
                  val_data_path=val_data_path)



def main(x_trn_path, x_val_path,
         n_train, d):
    """
    main function calling each element of the pipeline
    :param x_trn_path: path to directory with images to be used for training (str)
    directory must be structured as follows:

    xTrnDir
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
            ..
    :param x_val_path: path to directory with images to be used for validation (str) [optional]
            If provided, validation data directory must be structured like the training data directory.
    :param n_train: number of images to take for training (int)
    :param d: debugging mode [limit dataset size and training iterations] (int/bool)
    1=On, 0=Off
    :return: None
    """
    # set random seed for numpy and tensorflow
    seed(0)
    set_random_seed(0)

    if d == 1:
        print('debug mode: ON')
        n_train = 10
    print("iTrn: {}".format(x_trn_path))
    print("iVal: {}".format(x_val_path))
    print("n train: {}".format(n_train))
    print("debug mode: {}".format(bool(d)))

    """############################################################################
                        Preprocess Data
    ############################################################################"""
    # check if data path is valid
    data_path_dict = preprocess_data(x_trn_path, x_val_path, n_train)
    """############################################################################
                        Train
    ############################################################################"""
    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    has_val = x_val_path is not None
    train_models(models, data_path_dict, n_train, has_val, d)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main(args.x_trn_path,
             args.x_val_path,
             args.n_train,
             args.d)
        print('trnMeta.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
