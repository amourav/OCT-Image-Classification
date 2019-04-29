from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback
import sys
from PIL import Image
import numpy as np


def normImg(img):
    """
    normalize image intensity to zero mean and unit variance
    :param img (2d npy array): input image
    :return: normImg (2d npy array)
    """
    normImg = (img - np.mean(img)) / np.std(img)
    return normImg


def preprocessDir(dataPath,
                  outputPath,
                  subdir,
                  debug_,
                  newSize=(224,224)):
    """
    Preprocess directory of .jpeg images.
    Each image is normalized and resized to desired resolution
    :param dataPath (str): path to input directory of raw images
    :param outputPath (str): path to output directory
    :param subdir (str): preprocess either the train / test / val / all three subdirectories
    :param debug_ (int): debug_ = 1 to test code
    :param newSize (tuple (Xres, YRes)): desired resolution of preprocessed images
    :return: None
    """
    print(subdir)
    targetDataPath = os.path.join(dataPath, subdir)
    targetOutputPath = os.path.join(outputPath, subdir)
    diseaseDirs = os.listdir(targetDataPath)
    for imgType in diseaseDirs:
        print('\t', imgType)
        imgFilesPath = os.path.join(targetDataPath, imgType)
        if not(os.path.isdir(imgFilesPath)):
            continue
        imgFiles = os.listdir(imgFilesPath)
        imgFiles = [f for f in imgFiles if f.endswith('.jpeg')]
        sunDirOutputPath = os.path.join(targetOutputPath, imgType)
        if not(os.path.isdir(sunDirOutputPath)):
            os.makedirs(sunDirOutputPath)

        for imgFname in imgFiles:
            imgPath = os.path.join(imgFilesPath, imgFname)
            img = Image.open(imgPath) #np.asarray(Image.open(imgPath))
            imgReSized = img.resize(newSize, Image.ANTIALIAS) #skimage.transform.resize(img, newSize)
            imgArr = np.asarray(imgReSized)
            imgNorm = normImg(imgArr)
            imgFnameOut = imgFname.split('.jpeg')[0] + '.npy'
            imgOutPath = os.path.join(sunDirOutputPath, imgFnameOut)
            np.save(imgOutPath, imgNorm)
            if debug_:
                break


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
    module_parser.add_argument("-d", dest="d",
                               type=int,
                               default=0,
                               help='debug')
    return module_parser


def main_driver(dataPath, outputPath, subdir, d):
    """
    Initialize output directory and call preprocessDir

    :param dataPath (str): path to input directory of raw images
    :param outputPath (str): path to output directory
    :param subdir (str): preprocess either the train / test / val / all three subdirectories
    :param d (int): d = 1 to test code ()
    :return: None
    """
    if d == 1:
        debug_ = True
    else:
        debug_ = False
    assert(os.path.isdir(dataPath))
    if not(os.path.isdir(outputPath)):
        os.mkdir(outputPath)

    if subdir == 'all':
        for subdir in ['test', 'train', 'val']:
            preprocessDir(dataPath, outputPath, subdir, debug_)
    else:
        preprocessDir(dataPath, outputPath, subdir, debug_)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.dataPath,
                    args.outputPath,
                    args.subdir,
                    args.d)
        print('Done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
