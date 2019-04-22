from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
#from preprocessingUtils import normImg
import traceback
import sys
from PIL import Image
import numpy as np
#import logs.config_initialization as config_initialization


def normImg(img):
    normImg = (img - np.mean(img)) / np.std(img)
    return normImg


def preprocessDir(dataPath,
                  outputPath,
                  subdir,
                  debug_,
                  newSize=(224,224)):
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
            testImg = Image.open(imgPath)
            testImgReSized = testImg.resize(newSize, Image.ANTIALIAS)
            imgArr = np.asarray(testImgReSized)
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
                               default='debug',
                               help='debug')
    return module_parser


def main_driver(dataPath, outputPath, subdir, d):
    """

    :param dataPath:
    :param outputPath:
    :param subdir:
    :param debug_:
    :return:
    """
    if d == 1:
        debug_ = True
    else:
        debug_ = False
    assert(os.path.isdir(dataPath))
    if not(os.path.isdir(outputPath)):
        os.mkdirs(outputPath)

    if subdir == 'all':
        for subdir in ['test', 'train', 'val']:
            preprocessDir(dataPath, outputPath, subdir, debug_)
    else:
        targetDataPath = os.path.join(dataPath, subdir)
        targetOutputPath = os.path.join(outputPath, subdir)
        preprocessDir(dataPath, outputPath, subdir, debug_)

#%%

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
