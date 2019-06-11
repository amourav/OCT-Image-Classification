from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
import traceback
import sys
import numpy as np
import pandas as pd
from keras.backend import set_image_data_format
from keras.models import load_model
import skimage
from PIL import Image
sys.path.append('./code')
from trainCNN import getPreprocess


def preprocessImgs(xPath, newSize):
    """
    preprocess images in the directory and return data
    :param xPath: path to image directory containing .jpeg files (str)
    :param newSize: desired shape of each image array (tuple - [xRes, yRes, channels])
    :return: preprocessed image array (nImages, xRes, yRes, channels), list of image names
    """
    imgFiles = os.listdir(xPath)
    imgFiles = [f for f in imgFiles if f.endswith('.jpeg')]
    imgStack = []
    for imgFname in imgFiles:
        imgPath = os.path.join(xPath, imgFname)
        imgArr = np.array(Image.open(imgPath))
        imgArr = skimage.transform.resize(imgArr, newSize)
        imgArr = imgArr/imgArr.max()
        imgStack.append(imgArr)
    imgStack = np.stack(imgStack, axis=0)
    imgNames = [n.split('.')[0] for n in imgFiles]  # include text before .jpeg file ending
    return imgStack, imgNames


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i", dest="xPath", type=str,
                               help="input image directory")
    module_parser.add_argument("-o", dest="outPath", type=str,
                               help="output dir path")
    module_parser.add_argument("-m", dest="modelPath",
                               type=str,
                               help="path to keras model .hdf5 file")
    return module_parser


def main(xPath, outPath, xRes, yRes, modelPath):
    """
    takes a directory of new images and runs inference with the trained CNN
    :param xPath: path to directory of new images
    :param outPath: path to directory for predictions
    :param xRes: desired image width for preprocessing [may be changed for model] (int)
    :param yRes: desired image height for preprocessing [may be changed for model] (int)
    :param modelPath: path to .hdf5 file of trained CNN
    :return: None
    """

    """############################################################################
                        0. Preprocess Data
    ############################################################################"""
    imgTypeDict = {
        0: "NORMAL",
        1: "DRUSEN",
        2: "CNV",
        3: "DME",
    }

    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    # chose which preprocessing method to use by detecting the
    # model name in modelPath
    idxList = []
    for model in models:
        i = model in modelPath
        idxList.append(i)
    assert(sum(idxList)==1)
    modelName = models[np.argmax(idxList)]
    preprocessInput = getPreprocess(modelName)

    if modelName == "VGG16" or modelName == "ResNet50":
        xRes, yRes = 224, 224
    elif modelName == "Xception" or modelName == "InceptionV3":
        xRes, yRes = 299, 299
    newSize = (xRes, yRes, 3)
    imgData, imgNames = preprocessImgs(xPath, newSize)

    if preprocessInput is not None:
        imgData = preprocessInput(imgData)

    """############################################################################
                        0. load model & predict
    ############################################################################"""
    set_image_data_format('channels_last')
    model = load_model(modelPath)
    yPred = model.predict(imgData,
                          batch_size=1,
                          verbose=1)
    # save predictions to csv
    cols = []
    for i in range(yPred.shape[1]):
        cols.append(modelName + "_{}_{}".format(imgTypeDict[i], i))
    yPredDf = pd.DataFrame(yPred,
                           columns=cols,
                           index=imgNames)
    yPredDf.to_csv(join(outPath, "{}_predictions.csv".format(modelName)))


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main(args.xPath,
             args.outPath,
             args.modelPath)
        print('predict.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
