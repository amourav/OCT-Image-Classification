from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import sys
import os
from os.path import join
import traceback
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
sys.path.append('./code')
from evalUtils import UrgentVRoutne, reportBinaryScores


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-yTrue", dest="truePath", type=str,
                               help="path to yTrue csv")
    module_parser.add_argument("-yPred", dest="predPath", type=str,
                               help="output dir path csv")
    return module_parser


def main(truePath, predPath):
    assert(os.path.isfile(truePath))
    assert (os.path.isfile(predPath))
    yTrueDF = pd.read_csv(truePath, index_col=0)
    yPredDF = pd.read_csv(predPath, index_col=0)
    yDF = pd.merge(yTrueDF, yPredDF, left_index=True, right_index=True)
    yTrue = yDF["test"].values
    yPred = yDF[yPredDF.columns].values

    classMap = {
        "NORMAL": 0,
        "DRUSEN": 1,
        "CNV": 2,
        "DME": 3}

    #evaluate
    yTrue1Hot = to_categorical(yTrue)
    yTrueTestUrgent = UrgentVRoutne(yTrue1Hot, classMap).astype(np.int)
    classAcc = accuracy_score(yTrue,
                              yPred.argmax(axis=1))
    print('\t accuracy: {0:.3g}'.format(classAcc))
    yTestPredUrgent = UrgentVRoutne(yPred, classMap)
    print('\t binary (urgent vs non-urgent)')
    scores = reportBinaryScores(yTrueTestUrgent, yTestPredUrgent, v=1)
    acc, tpr, tnr, plr, nlr = scores
    fprs, tprs, _ = roc_curve(yTrueTestUrgent, yTestPredUrgent)
    aucUrgent = auc(fprs, tprs)
    outputPath = os.path.dirname(predPath)
    with open(os.path.join(outputPath, 'eval.txt'), 'w') as fid:
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
        main(args.truePath,
             args.predPath)
        print('evaluate.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
