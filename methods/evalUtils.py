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
from keras.backend import set_image_data_format

set_image_data_format('channels_last')


def reportBinaryScores(yTrueUrgent, yPredProbUrgent, v=0):
    yPredUrgent = yPredProbUrgent.round().astype(np.int)
    tn, fp, fn, tp = confusion_matrix(yTrueUrgent.astype(np.float),
                                      yPredUrgent).ravel()
    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    fpr = fp/(fp + tn)
    fnr = fn/(fn + tp)
    plr = tpr/fpr #positive likelihood ratio
    nlr = fnr/tnr # negative likelihood ratio
    acc = accuracy_score(yTrueUrgent,
                         yPredUrgent)
    if v:
        print('\t accuracy: {0:.3g}'.format(acc))
        print("\t sensitivity {0:.3g}".format(tpr))
        print("\t specificity {0:.3g}".format(tnr))
        print("\t positive likelihood ratio {0:.3g}".format(plr))
        print("\t negative likelihood ratio {0:.3g}".format(nlr))

    return acc, tpr, tnr, plr, nlr


def plotModelHist(modelHistory,
                  lossName='categorical cross entropy',
                  metricName='acc',
                  show=True):
    """
    plot the loss history and additional metric
    input: modelHistory (output of model.fit)
    return: matplotlib figure
    """
    hist = modelHistory #.history
    trnLoss = np.array(hist['loss'])
    valLoss = np.array(hist['val_loss'])
    trnMetric = np.array(hist[metricName])
    valMetic = np.array(hist['val_'+metricName])

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(trnLoss, c='r', label='train')
    plt.plot(valLoss, c='b', label='val')
    plt.title('loss ({})'.format(lossName))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(trnMetric, c='r', label='train')
    plt.plot(valMetic, c='b', label='val')
    plt.title(metricName)
    plt.legend()
    if show:
        plt.show()
    return fig


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
    :param y: target vector
    """
    catList = [] 
    for yi in y:
        catList.append(classMap[yi])
    return(np.array(catList))


def getROC(yTrue, yPred, classMap):
    assert(yTrue.shape==yPred.shape)
    assert(len(yTrue.shape)==2)
    yTrue1Hot, yPred1Hot = yTrue, yPred
    #yTrue, yPred = oneHotDecoder(yTrue), oneHotDecoder(yPred)
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
    classMapR = {i:lbl for lbl,i in classMap.items()}
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
    assert(yTrue.shape==yPred.shape)
    assert(len(yTrue.shape)==2)
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
        #True Positives
        tp = cfDF.loc[lbl, lbl]
        #false negatives
        fn = cfDF.loc[lbl][cfDF.columns!=lbl].sum()
        #false positives
        fp = cfDF[lbl][cfDF.columns!=lbl].sum()
        #true negatives
        tn = cfDF.loc[cfDF.index!=lbl, cfDF.columns!=lbl].sum().sum()
        tpr = tp/(tp + fn)
        tnr = tn/(tn + fp)
        fpr = fp/(fp + tn)
        fnr = fn/(fn + tp)
        plr = tpr/fpr #positive likelihood ratio
        nlr = fnr/tnr # negative likelihood ratio
        print("\t sensitivity {0:.3g}".format(tpr))
        print("\t specificity {0:.3g}".format(tnr))
        print("\t positive likelihood ratio {0:.3g}".format(plr))
        print("\t negative likelihood ratio {0:.3g}".format(nlr))
        print("\n")
    
    # Urgent Vs NonUrgent
    print('Urgent vs Non-Urgent')
    classMapR = {i:lbl for lbl,i in classMap.items()}
    yTrueUrgent = UrgentVRoutne(yTrue1Hot, classMapR).astype(np.int)
    yPredProbUrgent = UrgentVRoutne(yPred1Hot, classMapR)
    yPredUrgent = yPredProbUrgent.round().astype(np.int)
    tn, fp, fn, tp = confusion_matrix(yTrueUrgent.round(), 
                                      yPredUrgent).ravel()
    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    fpr = fp/(fp + tn)
    fnr = fn/(fn + tp)
    plr = tpr/fpr  # positive likelihood ratio
    nlr = fnr/tnr  # negative likelihood ratio
    acc = accuracy_score(yTrueUrgent, 
                         yPredUrgent)
    err = 1 - acc
    print('\t accuracy: {0:.3g}'.format(acc))
    print('\t error: {0:.3g}'.format(err))
    print("\t sensitivity {0:.3g}".format(tpr))
    print("\t specificity {0:.3g}".format(tnr))
    print("\t positive likelihood ratio {0:.3g}".format(plr))
    print("\t negative likelihood ratio {0:.3g}".format(nlr))
    print("\n"*6)
    return cfDF
