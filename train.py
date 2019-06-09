"""
@authors: Andrei Mouraviev & Eric TK Chou
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
import traceback
import sys
import numpy as np
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
    module_parser.add_argument("-Rx", dest="xRes", type=int, default=299,
                               help='x resulution for final img')
    module_parser.add_argument("-Ry", dest="yRes", type=int, default=299,
                               help='y resolution of final image')
    module_parser.add_argument("-d", dest="d", type=int,
                               default=0, help='debug mode')
    return module_parser


def main(xTrnPath, xValPath,
         modelName, modelWeights,
         nTrain, xRes, yRes, d):

    """############################################################################
                        0. Preprocess Data
    ############################################################################"""
    if d == 1:
        print('debug mode: ON')
        nTrain = 10

    assert(os.path.isdir(xTrnPath))
    if xValPath is not None:
        assert(os.path.isdir(xValPath))
    newSize = (int(xRes), int(yRes), 3)
    outputPath = "./PreprocessedData/{}".format(str(newSize))
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    preprocessDir(xTrnPath, outputPath,
                  'train', nTrain, newSize)

    if xValPath is not None:
        preprocessDir(xValPath, outputPath,
                      'val', nTrain, newSize)

    """############################################################################
                        1. Load Data
    ############################################################################"""
    trnTag = "{}_{}".format(str(newSize), 'train')
    trnDataPath = join(outputPath, "imgData_{}_n{}.npy".format(trnTag, nTrain))
    trnTargetPath = join(outputPath, "targetData_{}.npy".format(trnTag))
    xTrn = np.load(trnDataPath)
    yTrn = np.load(trnTargetPath)

    if xValPath is not None:
        valTag = "{}_{}".format(str(newSize), 'val')
        valDataPath = join(outputPath, "imgData_{}.npy".format(valTag))
        valTargetPath = join(outputPath, "targetData_{}.npy".format(valTag))
        XVal = np.load(valDataPath)
        yVal = np.load(valTargetPath)
    else:
        XVal = None
        yVal = None

    """############################################################################
                        1. Train CNN
    ############################################################################"""

    modelOutputPath = "./modelOutput/{}".format(modelName)
    if not os.path.isdir(modelOutputPath):
        os.makedirs(modelOutputPath)

    modelOutputDir = trainModel(xTrn, yTrn,
                                XVal, yVal,
                                modelOutputPath,
                                modelName, modelWeights,
                                aug=0, d=d, note="", xTest=None)

    with open(os.path.join(modelOutputDir, 'dataInfo.txt'), 'w') as fid:
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
             args.xRes,
             args.yRes,
             args.d)
        print('train.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
