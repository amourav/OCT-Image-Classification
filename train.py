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
from preprocess import preprocessDir
from trainCNN import trainModel


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-iTrn", dest="xTrnPath", type=str,
                               help="training data directory")
    module_parser.add_argument("-iVal", dest="xValPath",
                               type=str, default=None,
                               help="validation data directory")
    module_parser.add_argument("-n", dest="nTrain", type=int, default=1000,
                               help='number of images for training')
    module_parser.add_argument("-d", dest="d", type=int,
                               default=0, help='debug mode')
    return module_parser


def preprocessData(xTrnPath, xValPath, nTrain):
    """
    preliminary preprocessing to noralize image size and intensity
    preprocess for 224x224 and 299x299 resolutions
    :param xTrnPath: path to training data (str)
    :param xValPath: path to val data (str or None)
    :param nTrain: number of training samples (int)
    :return: dictionary of paths to preprocessed data indexed by resolution (dict)
    """
    assert (os.path.isdir(xTrnPath))
    if xValPath is not None:
        assert (os.path.isdir(xValPath))

    # preprocess data for each type of network
    dataPathDict = {}
    for res in [224, 299]:
        newSize = (res, res, 3)
        outputDataPath = join(".", "PreprocessedData", str(newSize))
        if not os.path.isdir(outputDataPath):
            os.makedirs(outputDataPath)
        preprocessDir(xTrnPath, outputDataPath,
                      'train', nTrain, newSize)
        if xValPath is not None:
            preprocessDir(xValPath, outputDataPath,
                          'val', nTrain, newSize)
        dataPathDict[res] = outputDataPath
    return dataPathDict

def saveInfo(outputModelPath,
             trnDataPath,
             valDataPath=None):
    """
    save training info
    :param outputModel Path: write path (str)
    :param trnDataPath: path to training data  (str)
    :param valDataPath: path to validation data (str)
    :return: None
    """

    with open(os.path.join(outputModelPath, 'dataInfo.txt'), 'w') as fid:
        fid.write("XTrainPath: {} \n".format(trnDataPath))
        if valDataPath is not None:
            fid.write("XValPath: {} \n".format(valDataPath))
        else:
            fid.write("XValPath: {} \n".format(trnDataPath))


def trainModels(models, dataPathDict, nTrain, hasVal, d):
    """
    Train each model in models
    :param models: list of model names (list)
    :param dataPathDict: dict containing path to data (dict)
    :param nTrain: number of training samples to use (int)
    :param hasVal: seperate validatio data present (bool)
    :param d: debug mode 1=ON, 0=OFF (int/bool)
    :return: None
    """
    now = datetime.datetime.now()
    today = str(now.date())
    for modelName in models:
        if modelName == "VGG16":
            res = 224
        else:
            res = 299
        # load data for each network
        outputDataPath = dataPathDict[res]
        print('loading data')
        trnTag = "{}_{}".format(str((res, res, 3)), 'train')
        trnDataPath = join(outputDataPath, "imgData_{}_n{}.npy".format(trnTag, nTrain))
        trnTargetPath = join(outputDataPath, "targetData_{}.npy".format(trnTag))
        xTrn = np.load(trnDataPath)
        yTrn = np.load(trnTargetPath)

        if hasVal:
            valTag = "{}_{}".format(str((res, res, 3)), 'val')
            valDataPath = join(outputDataPath, "imgData_{}.npy".format(valTag))
            valTargetPath = join(outputDataPath, "targetData_{}.npy".format(valTag))
            XVal = np.load(valDataPath)
            yVal = np.load(valTargetPath)
        else:
            valDataPath = None
            XVal = None
            yVal = None

        # save each CNN in a subdirectory
        outputModelPath = "./modelOutput/metaClf_{}/{}".format(today,
                                                               modelName)
        if not os.path.isdir(outputModelPath):
            os.makedirs(outputModelPath)
        trainModel(xTrn, yTrn,
                   XVal, yVal,
                   outputModelPath,
                   modelName, 'imagenet',
                   aug=0, d=d, note="", xTest=None)
        # save details of training in each CNN directory
        saveInfo(outputModelPath,
                 trnDataPath,
                 valDataPath=valDataPath)



def main(xTrnPath, xValPath,
         nTrain, d):
    """
    main function calling each element of the pipeline
    :param xTrnPath: path to directory with images to be used for training (str)
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
    :param xValPath: path to directory with images to be used for validation (str) [optional]
            If provided, validation data directory must be structured like the training data directory.
    :param nTrain: number of images to take for training (int)
    :param d: debugging mode [limit dataset size and training iterations] (int/bool)
    1=On, 0=Off
    :return: None
    """
    # set random seed for numpy and tensorflow
    seed(0)
    set_random_seed(0)

    if d == 1:
        print('debug mode: ON')
        nTrain = 10
    print("iTrn: {}".format(xTrnPath))
    print("iVal: {}".format(xValPath))
    print("n train: {}".format(nTrain))
    print("debug mode: {}".format(bool(d)))

    """############################################################################
                        Preprocess Data
    ############################################################################"""
    # check if data path is valid
    dataPathDict = preprocessData(xTrnPath, xValPath, nTrain)
    """############################################################################
                        Train
    ############################################################################"""
    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    hasVal = xValPath is not None
    trainModels(models, dataPathDict, nTrain, hasVal, d)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main(args.xTrnPath,
             args.xValPath,
             args.nTrain,
             args.d)
        print('trnMeta.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
