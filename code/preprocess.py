from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback
import sys
from PIL import Image
import numpy as np
import pandas as pd
import skimage


def normImg(img):
    """
    normalize image intensity to zero mean and unit variance
    :param img(2d npy array): input image
    :return: normImg (2d npy array)
    """
    normImg = (img - np.mean(img)) / np.std(img)
    return normImg


def preprocessDir(dataPath,
                  outputPath,
                  dataset,
                  nTrain,
                  newSize,
                  ):
    """
    Preprocess directory of .jpeg images.
    Each image is normalized and resized to desired resolution
    Final data is a numpy stack of image files compatible with
    keras model fit
    :param dataPath (str): base dir of raw data
    :param outputPath (str):
    :param dataset (str):
    :param debug_ (int/bool):
    :param newSize (tuple): resolution of final img
    :return:
    """

    imgTypeDict = {
        "NORMAL": 0,
        "DRUSEN": 1,
        "CNV": 2,
        "DME": 3,
    }
    print('Preprocessing:', dataset)
    targetDataPath = dataPath
    diseaseDirs = os.listdir(targetDataPath)
    diseaseDirs = [d for d in diseaseDirs if
                   os.path.isdir(os.path.join(targetDataPath, d))]
    imgStack, targetList = [], []
    imgNames = []
    for imgType in diseaseDirs:
        classLbl = imgTypeDict[imgType]
        nClass = int(nTrain / len(diseaseDirs))
        print('\t', imgType)
        imgFilesPath = os.path.join(targetDataPath, imgType)
        if not (os.path.isdir(imgFilesPath)):
            continue
        imgFiles = os.listdir(imgFilesPath)
        imgFiles = [f for f in imgFiles if f.endswith('.jpeg')]
        if dataset == 'train':
            imgFiles = np.random.choice(imgFiles, nClass)
        for imgFname in imgFiles:
            imgPath = os.path.join(imgFilesPath, imgFname)
            imgArr = np.array(Image.open(imgPath))
            imgArr = skimage.transform.resize(imgArr, newSize)
            #imgArr = normImg(imgArr) #!!
            imgStack.append(imgArr)
            targetList.append(classLbl)
        imgNames += [n.split('.')[0] for n in imgFiles]
    imgStack = np.stack(imgStack, axis=0)
    targetList = np.asarray(targetList)
    targetDF = pd.DataFrame(index=imgNames)
    targetDF[dataset] = targetList
    infoTag = "{}_{}".format(str(newSize), dataset)
    if dataset == 'train':
        imgStackOutPath = os.path.join(outputPath, "imgData_{}_n{}.npy".format(infoTag,
                                                                               nTrain))
    else:
        imgStackOutPath = os.path.join(outputPath, "imgData_{}.npy".format(infoTag))
    targetListOutPath = os.path.join(outputPath, "targetData_{}.npy".format(infoTag))
    np.save(imgStackOutPath, imgStack)
    np.save(targetListOutPath, targetList)
    targetDF.to_csv(os.path.join(outputPath, "targetData_{}.csv".format(infoTag)))


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
    module_parser.add_argument("-Rx", dest="xRes", type=int,
                               help='x resulution for final img')
    module_parser.add_argument("-Ry", dest="yRes", type=int,
                               help='y resolution of final image')
    module_parser.add_argument("-d", dest="d",
                               type=int,
                               default=0,
                               help='debug')
    return module_parser


def main_driver(dataPath, outputPath, subdir,
                nTrain, xRes, yRes, d):
    """
    Initialize output directory and call preprocessDir

    :param dataPath (str): path to input directory of raw images
    :param outputPath (str): path to output directory
    :param subdir (str): preprocess either the train / test / val / all three subdirectories
    :param d (int): d = 1 to test code ()
    :return: None
    """
    if d == 1:
        print('debug mode: ON')
        subdir = 'train'
        nTrain = 10

    assert(os.path.isdir(dataPath))
    newSize = (int(xRes), int(yRes), 3)
    if not(os.path.isdir(outputPath)):
        os.makedirs(outputPath)
    print(outputPath)
    if subdir == 'all':
        for subdir in ['test', 'train', 'val']:
            preprocessDir(os.path.join(dataPath, subdir),
                          outputPath, subdir, nTrain, newSize)
    else:
        preprocessDir(os.path.join(dataPath, subdir),
                      outputPath, subdir, nTrain, newSize)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.dataPath,
                    args.outputPath,
                    args.subdir,
                    args.nTrain,
                    args.xRes,
                    args.yRes,
                    args.d)
        print('Done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
