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
    module_parser.add_argument("-m", dest="modelName",
                               type=str, default="Xception",
                               help="model name (default - Xception)")
    module_parser.add_argument("-w", dest="modelWeights",
                               type=str, default='imagenet',
                               help="path to model weights")
    module_parser.add_argument("-n", dest="nTrain", type=int, default=1000,
                               help='n: number of images for training')
    module_parser.add_argument("-d", dest="d", type=int,
                               default=0, help='debug mode')
    return module_parser


def main(xTrnPath, xValPath,
         modelName, modelWeights,
         nTrain, d):
    """
    main function calling each element of the pipeline
    :param xTrnPath: path to directory with images to be used for training (str)
    :param xValPath: path to directory with images to be used for validation (str) [optional]
    :param modelName: architecture name (str)
    :param modelWeights: path to pre-trained network weights
    :param nTrain: number of images to take for training
    :param d: debugging mode [limit dataset size and training iterations] (int/bool)
    1=On, 0=Off
    :return: None
    """

    """############################################################################
                        0. Preprocess Data
    ############################################################################"""
    # set random seed for numpy and tensorflow
    seed(0)
    set_random_seed(0)

    if d == 1:
        print('debug mode: ON')
        nTrain = 10

    assert(os.path.isdir(xTrnPath))
    if xValPath is not None:
        assert(os.path.isdir(xValPath))
    if modelName == "VGG16":
        xRes, yRes = 224, 224
    elif modelName in ["Xception", "InceptionV3", "ResNet50"]:
        xRes, yRes = 299, 299
    newSize = (int(xRes), int(yRes), 3)
    outputDataPath = "./PreprocessedData/{}".format(str(newSize))
    if not os.path.isdir(outputDataPath):
        os.makedirs(outputDataPath)
    preprocessDir(xTrnPath, outputDataPath,
                  'train', nTrain, newSize)
    if xValPath is not None:
        preprocessDir(xValPath, outputDataPath,
                      'val', nTrain, newSize)

    """############################################################################
                        1. Load Data
    ############################################################################"""
    print('loading data')
    trnTag = "{}_{}".format(str(newSize), 'train')
    trnDataPath = join(outputDataPath, "imgData_{}_n{}.npy".format(trnTag, nTrain))
    trnTargetPath = join(outputDataPath, "targetData_{}.npy".format(trnTag))
    xTrn = np.load(trnDataPath)
    yTrn = np.load(trnTargetPath)

    if xValPath is not None:
        valTag = "{}_{}".format(str(newSize), 'val')
        valDataPath = join(outputDataPath, "imgData_{}.npy".format(valTag))
        valTargetPath = join(outputDataPath, "targetData_{}.npy".format(valTag))
        XVal = np.load(valDataPath)
        yVal = np.load(valTargetPath)
    else:
        XVal = None
        yVal = None

    """############################################################################
                        1. Train CNN
    ############################################################################"""
    now = datetime.datetime.now()
    today = str(now.date()) + \
                '_' + str(now.hour) + \
                '_' + str(now.minute)
    outputModelPath = "./modelOutput/{}".format(modelName)
    outputModelPath = outputModelPath + '_' + today
    if not os.path.isdir(outputModelPath):
        os.makedirs(outputModelPath)

    trainModel(xTrn, yTrn,
               XVal, yVal,
               outputModelPath,
               modelName, modelWeights,
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
             args.modelName,
             args.modelWeights,
             args.nTrain,
             args.d)
        print('train.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
