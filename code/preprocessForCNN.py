from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
#from preprocessingUtils import normImg
import traceback
import sys
from PIL import Image
import numpy as np


def preprocessForCNN(dataPath, outputPath, subdir, nTrain, debug_):
    """

    :param dataPath:
    :param outputPath:
    :param subdir:
    :param nTrain:
    :param debug_:
    :return:
    """
    imgTypeDict = {
        "NORMAL": 0,
        "DRUSEN": 1,
        "CNV": 2,
        "DME": 3,
    }

    print(subdir)
    targetDataPath = os.path.join(dataPath, subdir)

    diseaseDirs = os.listdir(targetDataPath)
    imgStack, targetList = [], []
    for imgType in diseaseDirs:
        classLbl = imgTypeDict[imgType]
        nClass = int(nTrain/len(diseaseDirs))
        print('\t', imgType)
        imgFilesPath = os.path.join(targetDataPath, imgType)
        if not(os.path.isdir(imgFilesPath)):
            continue
        imgFiles = os.listdir(imgFilesPath)
        imgFiles = [f for f in imgFiles if f.endswith('.npy')]
        if subdir == 'train':
            imgFiles = np.random.choice(imgFiles, nClass)

        for imgFname in imgFiles:
            imgPath = os.path.join(imgFilesPath, imgFname)
            imgArr = np.load(imgPath)
            imgStack.append(imgArr)
            targetList.append(classLbl)

    imgStack = np.stack(imgStack, axis=0)
    targetList = np.asarray(targetList)

    #shuffle
    idxList = np.arange(len(targetList))
    np.random.shuffle(idxList)
    imgStack = imgStack[idxList]
    targetList = targetList[idxList]

    if subdir == 'train':
        imgStackOutPath = os.path.join(outputPath, "imgData_{}_{}.npy".format(subdir, nTrain))
    else:
        imgStackOutPath = os.path.join(outputPath, "imgData_{}.npy".format(subdir))
    targetListOutPath = os.path.join(outputPath, "targetData_{}.npy".format(subdir))

    np.save(imgStackOutPath, imgStack)
    np.save(targetListOutPath, targetList)


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i", dest="dataPath", type=str,
                               help="the location dataset")
    module_parser.add_argument("-o", dest="outputPath", type=str,
                               help='base dir for outputs')
    module_parser.add_argument("-subdir", dest="subdir", type=str,
                               choices=['test', 'train', 'val', 'all'],
                               help='subdir: trn, test, val, or all ...')
    module_parser.add_argument("-n", dest="nTrain", type=int,
                               help='n: number of images for training')
    module_parser.add_argument("-d", dest="d",
                               type=int,
                               default=0,
                               help='debug')
    return module_parser


def main_driver(dataPath, outputPath, subdir, nTrain, d):
    """

    :param dataPath:
    :param outputPath:
    :param subdir:
    :param debug_:
    :return:
    """
    if d == 1:
        print('Debugging mode ON')
        debug_ = True
        nTrain = 10
    else:
        debug_ = False
    assert(os.path.isdir(dataPath))
    if not(os.path.isdir(outputPath)):
        os.mkdir(outputPath)

    if subdir == 'all':
        for subdir in ['test', 'train', 'val']:
            preprocessForCNN(dataPath, outputPath, subdir, nTrain, debug_)
    else:
        preprocessForCNN(dataPath, outputPath, subdir, nTrain, debug_)

#%%

if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.dataPath,
                    args.outputPath,
                    args.subdir,
                    args.nTrain,
                    args.d)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
