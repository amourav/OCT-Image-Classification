from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback
import sys
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from keras.applications.vgg16 import preprocess_input as preprocess_input_xception
from keras.utils import to_categorical
import keras.backend as K
from keras.backend import set_image_data_format
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from evalUtils import UrgentVRoutne, reportBinaryScores


def getPreprocess(modelName):
    if modelName == 'InceptionV3':
        preprocessInput = preprocess_input_inception_v3
    elif modelName == 'VGG16':
        preprocessInput = None #preprocess_input_vgg16
    elif modelName == 'ResNet50':
        preprocessInput = preprocess_input_ResNet50
    elif modelName == 'Xception':
        preprocessInput = preprocess_input_xception
    else:
        raise Exception('model name not recognized')
    return preprocessInput


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-x", dest="XTestPath", type=str,
                               help='x test path')
    module_parser.add_argument("-y", dest="yTestPath", type=str,
                               default=None,
                               help='model weights')
    module_parser.add_argument("-m", dest="modelPath", type=str,
                               help='model path')
    module_parser.add_argument("-n", dest="note",
                               type=str,
                               default='',
                               help='note: will be added to output file path')
    return module_parser


def main_driver(XTestPath,
                yTestPath,
                modelPath,
                note):

    print('trn path:', XTestPath)
    set_image_data_format('channels_last')
    assert(os.path.isfile(XTestPath))
    assert(os.path.isfile(modelPath))
    if yTestPath is not None:
        assert(os.path.isfile(yTestPath))
        yTest = pd.read_csv(yTestPath, index_col=0)
        yTestArr = yTest.values
    XTest = np.load(XTestPath)
    outputPath = os.path.dirname(modelPath)
    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']

    idxList = []
    for model in models:
        i = model in modelPath
        idxList.append(i)
    assert(sum(idxList)==1)
    modelName = models[np.argmax(idxList)]
    preprocessInput = getPreprocess(modelName)
    if preprocessInput is not None:
        XTest = preprocessInput(XTest)
    model = load_model(modelPath)
    yTestPred = model.predict(XTest,
                              batch_size=32,
                              verbose=1)
    yTestPredPath = os.path.join(outputPath, 'yPred_{}.npy'.format(note))
    modelPredDF = pd.DataFrame(index=yTest.index)
    for yLbl in np.unique(yTest):
        modelPredDF[modelName + "_{}".format(yLbl)] = yTestPred[:, yLbl]
    modelPredDF['yTrueTest'] = yTest
    modelPredDF.to_csv(os.path.join(outputPath,
                                    'yPredDf_{}.csv'.format(note)))
    np.save(yTestPredPath, yTestPred)
    if yTest is not None:
        classMap = {
            "NORMAL": 0,
            "DRUSEN": 1,
            "CNV": 2,
            "DME": 3}
        yTrue1Hot = to_categorical(yTest)
        yTrueTestUrgent = UrgentVRoutne(yTrue1Hot, classMap).astype(np.int)
        classAcc = accuracy_score(yTest,
                                  yTestPred.argmax(axis=1))
        print('\t accuracy: {0:.3g}'.format(classAcc))
        yTestPredUrgent = UrgentVRoutne(yTestPred, classMap)
        print()
        print('\t binary (urgent vs non-urgent)')
        scores = reportBinaryScores(yTrueTestUrgent, yTestPredUrgent, v=1)
        acc, tpr, tnr, plr, nlr = scores
        fprs, tprs, _ = roc_curve(yTrueTestUrgent, yTestPredUrgent)
        aucUrgent = auc(fprs, tprs)
        with open(os.path.join(outputPath, 'eval_{}.txt'.format(note)), 'w') as fid:
            fid.write("model: {} \n".format(modelName))
            fid.write("4 classAcc: {} \n".format(classAcc))
            fid.write("{} \n".format('binary - urgent vs non-urgent'))
            fid.write("acc: {} \n".format(acc))
            fid.write("tpr: {} \n".format(tpr))
            fid.write("tnr: {} \n".format(tnr))
            fid.write("aucUrgent: {} \n".format(aucUrgent))

if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.XTestPath,
                    args.yTestPath,
                    args.modelPath,
                    args.note)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
