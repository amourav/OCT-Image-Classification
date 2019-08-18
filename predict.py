from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import traceback
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.backend import set_image_data_format
from keras.models import model_from_json
import skimage
from PIL import Image
sys.path.append('./methods')
from trainCNN import getPreprocess
from preprocess import getClassLabels


def getBinaryPred(modelPredDF):
    """
    Get probability of Urgent vs Non-Urgent diagnosis from predicted class label probabilities
    :param modelPredDF: Dataframe containing multi-class probabilities for each sample
    :return: Dataframe containing binary probabilities for each sample
    """
    urgentLabels = ['CNV', 'DME']
    urgentCols = []
    predUrgentDF = pd.DataFrame(index=modelPredDF.index)
    for col in modelPredDF.columns:
        if (urgentLabels[0] in col) or (urgentLabels[1] in col):
            urgentCols.append(col)
    assert(len(urgentCols) == 2)
    predUrgentDF['urgent_proba'] = modelPredDF[urgentCols[0]] + modelPredDF[urgentCols[1]]
    return predUrgentDF


def meanPrediction(modelPredDF, yVals=[0, 1, 2, 3]):
    """
    output the mean probability predicted for each class
    :param modelPredDF: dataframe containing the predictions
    from all models for each class (pandas Dataframe)
    :param yVals: possible output classes (list like object)
    :return: mean probability for each class (pandas dataframe)
    """
    imgTypeDict = getClassLabels(intKey=True)

    meanPred = pd.DataFrame(index=modelPredDF.index)
    for yi in np.unique(yVals):
        mean = modelPredDF.filter(regex='_{}'.format(yi)).mean(axis=1)
        meanPred['proba_{}'.format(imgTypeDict[yi])] = mean
    meanPred = meanPred.div(meanPred.sum(axis=1), axis=0)
    return meanPred


def loadModel(modelName, metaPath):
    """
    load keras model from json and model weights
    :param modelName: name of model (str)
    :param metaPath: path of directory containing
    all models of the ensemble classifier (str)
    :return: keras model, directory containing model files
    """
    modelDirs = os.listdir(metaPath)
    modelDirs = [d for d in modelDirs if os.path.isdir(join(metaPath, d))]

    # find path to individual model
    modelDir = None
    for dir in modelDirs:
        if modelName in dir:
            modelDir = dir

    modelDirPath = join(metaPath,
                        modelDir)
    modelWeightsPath = join(modelDirPath,
                            modelName + ".hdf5")
    # load json and create model
    print('loading {} json'.format(modelName))
    jsonPath = join(os.path.dirname(modelWeightsPath),
                    '{}_architecture.json'.format(modelName))
    jsonFile = open(jsonPath, 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    model = model_from_json(loadedModelJson)

    # load weights into new model
    model.load_weights(modelWeightsPath)
    print("Loaded {} json from disk".format(modelName))
    return model, modelDirPath


def preprocessImgs(xPath, newSize):
    """
    preprocess images in the directory and return data
    :param xPath: path to image directory containing .jpeg files (str)
    :param newSize: desired shape of each image array (tuple - [xRes, yRes, channels])
    :return: preprocessed image array (np array) [nImages, xRes, yRes, channels],
             list of image names (list)
    """
    imgFiles = os.listdir(xPath)
    imgFiles = [f for f in imgFiles if f.endswith('.jpeg')]
    imgStack = []
    for imgFname in imgFiles:
        imgPath = os.path.join(xPath, imgFname)
        imgArr = np.array(Image.open(imgPath))
        imgArr = skimage.transform.resize(imgArr, newSize)
        imgArr = (imgArr - imgArr.min())/imgArr.max()
        imgStack.append(imgArr)
    imgStack = np.stack(imgStack, axis=0)
    imgNames = [n.split('.')[0] for n in imgFiles]  # include text before .jpeg file ending
    return imgStack, imgNames


def saveModelResults(modelName, yPred, imgNames, modelDirPath, imgTypeDict):
    """
    save predictions of individual models
    :param modelName: name of model (e.g. VGG16) (str)
    :param yPred: model predictions (npy array)
    :param imgNames: list of image filenames (list)
    :param modelDirPath: path to the directory containing the model (str)
    :param imgTypeDict: mapping from integer labels to image labels (dict)
    :return: dataframe containing model predictions (pandas DF)
    """
    cols = ["{}_{}_{}".format(modelName, i, l)
            for (l, i)
            in imgTypeDict.items()]

    yPredDf = pd.DataFrame(yPred,
                           columns=cols,
                           index=imgNames)
    yPredDf.to_csv(join(modelDirPath,
                        "{}_predictions.csv".format(modelName)))
    return yPredDf


def savePredictions(modelPredDict, models, imgNames, outPath):
    """
    save the predictions of the ensemble classifier
    :param modelPredDict: dictionary containing the predictions of each model (dict)
    :param models: list of model names (list)
    :param imgNames: list of image filenames (list)
    :param outPath: directory where predictions are saved (str)
    :return:
    """
    # merge predictions into a single dataframe
    modelPredDF = pd.DataFrame(index=imgNames)
    for modelName in models:
        modelPredDF = pd.merge(modelPredDF,
                               modelPredDict[modelName],
                               left_index=True,
                               right_index=True)

    # calculate average probability for each class
    meanPredDF = meanPrediction(modelPredDF)
    binaryPredDF = getBinaryPred(meanPredDF)

    # save dataframes to csv
    modelPredDF.to_csv(join(outPath,
                            "individualModelPredictions.csv"))
    meanPredDF.to_csv(join(outPath,
                            "ensembleClfMeanProba.csv"))
    binaryPredDF.to_csv(join(outPath,
                             "urgentProba.csv"))


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i", dest="xPath", type=str,
                               help="input image directory")
    module_parser.add_argument("-o", dest="outPath", type=str,
                               help="output dir path")
    module_parser.add_argument("-m", dest="metaPath",
                               type=str,
                               help="path to model directory")
    return module_parser


def main(xPath, outPath, metaPath):
    """
    takes a directory of new images and runs inference with the classifier
    :param xPath: path to directory of new images (str)
    :param outPath: path to directory for predictions (str)
    :param metaPath: path to metaClf directory (str)
    :return: None
    """

    """############################################################################
                        0. Preprocess Data
    ############################################################################"""
    set_image_data_format('channels_last')
    imgTypeDict = {
        0: "NORMAL",
        1: "DRUSEN",
        2: "CNV",
        3: "DME",
    }

    modelPredDict = {}
    # generate predictions for each individual model
    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    for modelName in models:
        # chose which preprocessing method to use by detecting the model
        preprocessInput = getPreprocess(modelName)
        if modelName == "VGG16":
            res = 224
        else:
            res = 299
        newSize = (res, res, 3)
        imgData, imgNames = preprocessImgs(xPath, newSize)
        imgData = preprocessInput(imgData)

        """############################################################################
                            0. load model & predict
        ############################################################################"""
        # load model
        model, modelDirPath = loadModel(modelName, metaPath)
        # run inference
        yPred = model.predict(imgData,
                              batch_size=1,
                              verbose=1)
        # save intermediate predictions to csv
        yPredDf = saveModelResults(modelName, yPred,
                                   imgNames, modelDirPath,
                                   imgTypeDict)
        modelPredDict[modelName] = yPredDf
    savePredictions(modelPredDict, models, imgNames, outPath)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main(args.xPath,
             args.outPath,
             args.metaPath)
        print('predMeta.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
