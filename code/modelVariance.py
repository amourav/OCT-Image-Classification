import os
from os.path import join
import pickle
import sys
import numpy as np
import pandas as pd
from keras.backend import set_image_data_format
from keras.utils import to_categorical
from keras.models import load_model
from evalUtils import UrgentVRoutne
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import traceback


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-model", dest="modelBaseDir", type=str,
                               help="dir of sampling experiment")
    module_parser.add_argument("-data", dest="dataPath", type=str,
                               help="directory of test data")
    module_parser.add_argument("-o", dest="outputDir", type=str,
                               help="directory of output")
    module_parser.add_argument("-m", dest="modelName", type=str,
                               default='VGG16',
                               help='modelName (default: VGG16)')
    return module_parser


def main_driver(modelBaseDir, dataPath, outputDir, modelName):
    classMap = {
        "NORMAL": 0,
        "DRUSEN": 1,
        "CNV": 2,
        "DME": 3}

    classMapR = {i: lbl for lbl, i in classMap.items()}
    lbls = classMap.keys()

    varainceExperimentDir = modelBaseDir

    modelHistPathList = []
    bestModelPathList = []
    for i in range(1, 11):
        expPathi = join(varainceExperimentDir, str(i))
        files = os.listdir(expPathi)
        modelDir = [f for f in files if modelName in f]
        if not(len(modelDir) == 1):
          raise Exception(modelDir)
        modelPathi = join(expPathi, modelDir[0])
        modelHistPath = join(modelPathi, "modelHistory.pickle")
        modelPath = join(modelPathi, "{}.hdf5".format(modelName))
        modelHistPathList.append(modelHistPath)
        bestModelPathList.append(modelPath)
        assert (os.path.isfile(modelHistPath))
        assert (os.path.isfile(modelPath))

    xTestPath = join(dataPath, r"imgData_test.npy")
    yTestPath = join(dataPath, r"targetData_test.npy")

    xTest = np.load(xTestPath)
    yTest = np.load(yTestPath)

    yTrue1Hot = to_categorical(yTest)
    yTestUrgent = UrgentVRoutne(yTrue1Hot, classMap).astype(np.int)

    modelPredUrgetDF = pd.DataFrame()
    modelPredUrgetDF['yTrue'] = yTestUrgent

    yTestPredDict = {}
    for i, modelPath in enumerate(bestModelPathList):
        model = load_model(modelPath)
        yTestPred = model.predict(xTest, batch_size=20, verbose=1)
        yTestPredDict[i] = yTestPred
        yPredProbUrgent = UrgentVRoutne(yTestPred, classMap)
        modelPredUrgetDF['yPredProb_' + str(i)] = yPredProbUrgent
    with open(join(outputDir, 'modelsPredDict.pickle'), 'wb') as fid:
        pickle.dump(yTestPredDict, fid)
    modelPredUrgetDF.to_csv(join(outputDir, "modelPredUrgetDF.csv"))


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.modelBaseDir,
                    args.dataPath,
                    args.outputDir,
                    args.modelName)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
