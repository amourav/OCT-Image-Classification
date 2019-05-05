import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback


def oneHotDecoder(y):
    """
    Transform endpoints form one-hot encoding to integer labels.
    :param y: array, shape = [nSamples, nClasses]
    return y: integer labels, shape = [nSamples]
    """
    y = y.argmax(axis=1)
    return y


def int2Cat(y, classMap):
    """
    transform target array from integer labels to string labels
    :param y: target vector,
    """
    catList = []
    for yi in y:
        catList.append(classMapR[yi])
    return (np.array(catList))


def getROC(yTrue, yPred, classMap):
    assert (yTrue.shape == yPred.shape)
    assert (len(yTrue.shape) == 2)
    yTrue1Hot, yPred1Hot = yTrue, yPred
    # yTrue, yPred = oneHotDecoder(yTrue), oneHotDecoder(yPred)
    lbls = classMap.keys()
    TPRs, FPRs, AUCs = {}, {}, {}
    colors = ['red', 'green', 'blue', 'orange']
    plt.figure(figsize=(10, 10))
    for lbl in lbls:
        imgClass = classMap[lbl]
        FPRs[lbl], TPRs[lbl], _ = roc_curve(yTrue[:, lbl], yPred[:, lbl])
        AUCs[lbl] = auc(FPRs[lbl], TPRs[lbl])
        plt.plot(FPRs[lbl], TPRs[lbl], color=colors[lbl],
                 label='{} vs rest (auc={})'.format(imgClass,
                                                    round(AUCs[lbl], 2)))
    plt.plot([0, 1], [0, 1],
             'k--', label='chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()

    # Urgent Vs NonUrgent
    print('Urgent vs Non-Urgent')
    classMapR = {i: lbl for lbl, i in classMap.items()}
    yTrueUrgent = UrgentVRoutne(yTrue1Hot, classMapR).astype(np.int)
    yPredProbUrgent = UrgentVRoutne(yPred1Hot, classMapR)
    yPredUrgent = yPredProbUrgent.round().astype(np.int)
    plt.figure(figsize=(10, 10))
    fpr, tpr, _ = roc_curve(yTrueUrgent, yPredProbUrgent)
    aucUrgent = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='r', lw=2,
             label='Urgent vs Non-Urgent (auc={0:.3g})'.format(
                 aucUrgent))
    plt.plot([0, 1], [0, 1],
             'k--', label='chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()
    return AUCs


def UrgentVRoutne(y, classMap):
    urgentClasses = ["CNV", "DME"]
    urgentIdx = [classMap[imgClass] for imgClass in urgentClasses]
    urgentProbList = []
    for yi in y:
        urgentProb = yi[urgentIdx].sum()
        urgentProbList.append(urgentProb)
    return np.array(urgentProbList)


def getConfusionMatrix(yTrue, yPred, classMap, title='', plot=True):
    assert (yTrue.shape == yPred.shape)
    assert (len(yTrue.shape) == 2)
    yTrue1Hot, yPred1Hot = yTrue, yPred
    yTrue, yPred = oneHotDecoder(yTrue), oneHotDecoder(yPred)
    acc = accuracy_score(yTrue,
                         yPred)
    err = 1 - acc
    print('\t accuracy: {0:.3g}'.format(acc))
    print('\t error: {0:.3g}'.format(err))
    lbls = classMap.keys()
    cf = confusion_matrix(yTrue, yPred)
    cfDF = pd.DataFrame(cf, index=lbls, columns=lbls)
    if plot:
        sn.heatmap(cfDF, annot=True, cmap=sn.color_palette("Blues_r"))
        plt.xlabel('y Pred', fontsize=14)
        plt.ylabel('y True', fontsize=14)
        plt.title(title, fontsize=14)
        plt.show()

    N = len(yTrue)
    TPs, FNs, FPs, TNs = {}, {}, {}, {}
    sensitivity, specificity = {}, {}
    for lbl in lbls:
        imgClass = classMap[lbl]
        print(imgClass)
        # True Positives
        tp = cfDF.loc[lbl, lbl]
        # false negatives
        fn = cfDF.loc[lbl][cfDF.columns != lbl].sum()
        # false positives
        fp = cfDF[lbl][cfDF.columns != lbl].sum()
        # true negatives
        tn = cfDF.loc[cfDF.index != lbl, cfDF.columns != lbl].sum().sum()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        plr = tpr / fpr  # positive likelihood ratio
        nlr = fnr / tnr  # negative likelihood ratio
        print("\t sensitivity {0:.3g}".format(tpr))
        print("\t specificity {0:.3g}".format(tnr))
        print("\t positive likelihood ratio {0:.3g}".format(plr))
        print("\t negative likelihood ratio {0:.3g}".format(nlr))
        print("\n")

    # Urgent Vs NonUrgent
    print('Urgent vs Non-Urgent')
    classMapR = {i: lbl for lbl, i in classMap.items()}
    yTrueUrgent = UrgentVRoutne(yTrue1Hot, classMapR).astype(np.int)
    yPredProbUrgent = UrgentVRoutne(yPred1Hot, classMapR)
    yPredUrgent = yPredProbUrgent.round().astype(np.int)
    tn, fp, fn, tp = confusion_matrix(yTrueUrgent.round(),
                                      yPredUrgent).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    plr = tpr / fpr  # positive likelihood ratio
    nlr = fnr / tnr  # negative likelihood ratio
    acc = accuracy_score(yTrueUrgent,
                         yPredUrgent)
    err = 1 - acc
    print('\t accuracy: {0:.3g}'.format(acc))
    print('\t error: {0:.3g}'.format(err))
    print("\t sensitivity {0:.3g}".format(tpr))
    print("\t specificity {0:.3g}".format(tnr))
    print("\t positive likelihood ratio {0:.3g}".format(plr))
    print("\t negative likelihood ratio {0:.3g}".format(nlr))
    print("\n" * 6)
    return cfDF


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
    initialize output directory and call preprocessForCNN
    :param dataPath (str): path to preprocessed npy image arrays
    :param outputPath (str): path to output directory for preprocessed stacks of images
    :param subdir (str): preprocess either the train / test / val / all three subdirectories
    :param nTrain (int): random sample training
    :param d: set d=1 for code testing.
    :return:
    """
    if not(type(nTrain)==int):
        nTrain = int(nTrain)

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





