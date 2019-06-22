"""
@authors: Andrei Mouraviev & Eric TK Chou
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
import traceback
import sys
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import datetime
sys.path.append('./code')
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
                               help='n: number of images for training')
    module_parser.add_argument("-d", dest="d", type=int,
                               default=0, help='debug mode')
    return module_parser


def main(xTrnPath, xValPath,
         nTrain, d):
    """
    main function calling each element of the pipeline
    :param xTrnPath: path to directory with images to be used for training (str)
    :param xValPath: path to directory with images to be used for validation (str) [optional]
    :param nTrain: number of images to take for training
    :param d: debugging mode [limit dataset size and training iterations] (int/bool)
    1=On, 0=Off
    :return: None
    """
    # set random seed for numpy and tensorflow
    seed(0)
    set_random_seed(0)
    now = datetime.datetime.now()
    today = str(now.date())

    if d == 1:
        print('debug mode: ON')
        nTrain = 10
    print("n train: {}".format(nTrain))

    """############################################################################
                        0. Preprocess Data
    ############################################################################"""
    assert(os.path.isdir(xTrnPath))
    if xValPath is not None:
        assert(os.path.isdir(xValPath))

    dataPathDict = {}
    for res in [224, 299]:
        newSize = (res, res, 3)
        outputDataPath = "./PreprocessedData/{}".format(str(newSize))
        if not os.path.isdir(outputDataPath):
            os.makedirs(outputDataPath)
        preprocessDir(xTrnPath, outputDataPath,
                      'train', nTrain, newSize)
        if xValPath is not None:
            preprocessDir(xValPath, outputDataPath,
                          'val', nTrain, newSize)
        dataPathDict[res] = outputDataPath

    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    for modelName in models:
        if modelName == "VGG16":
            res = 224
        else:
            res = 299

        # LOAD DATA
        outputDataPath = dataPathDict[res]
        print('loading data')
        trnTag = "{}_{}".format(str((res, res, 3)), 'train')
        trnDataPath = join(outputDataPath, "imgData_{}_n{}.npy".format(trnTag, nTrain))
        trnTargetPath = join(outputDataPath, "targetData_{}.npy".format(trnTag))
        xTrn = np.load(trnDataPath)
        yTrn = np.load(trnTargetPath)

        if xValPath is not None:
            valTag = "{}_{}".format(str((res, res, 3)), 'val')
            valDataPath = join(outputDataPath, "imgData_{}.npy".format(valTag))
            valTargetPath = join(outputDataPath, "targetData_{}.npy".format(valTag))
            XVal = np.load(valDataPath)
            yVal = np.load(valTargetPath)
        else:
            XVal = None
            yVal = None

        # TRAIN CNN
        outputModelPath = "./modelOutput/metaClf_{}/{}".format(today,
                                                               modelName)
        if not os.path.isdir(outputModelPath):
            os.makedirs(outputModelPath)

        trainModel(xTrn, yTrn,
                   XVal, yVal,
                   outputModelPath,
                   modelName, 'imagenet',
                   aug=0, d=d, note="", xTest=None)

        with open(os.path.join(outputModelPath, 'dataInfo.txt'), 'w') as fid:
            fid.write("XTrainPath: {} \n".format(trnDataPath))
            fid.write("XValPath: {} \n".format(valDataPath))


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
