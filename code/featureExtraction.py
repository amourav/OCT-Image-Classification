from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
import traceback
import sys
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.backend import set_image_data_format


def getModelFeatures(X, modelName, weights):
    if modelName == 'InceptionV3':
        base_model = InceptionV3(weights=weights,
                                 include_top=False)
    elif modelName == 'VGG16':
        base_model = VGG16(weights=weights,
                           include_top=False)

    elif modelName == 'ResNet50':
        base_model = ResNet50(weights=weights,
                              include_top=False)
    elif modelName == 'Xception':
        base_model = Xception(weights=weights,
                              include_top=False)
    else:
        raise Exception('model name not recognized')

    modelFeatures = base_model.predict(X, batch_size=1)
    return modelFeatures

def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-x", dest="XPath", type=str,
                               help="X  path ")
    module_parser.add_argument("-o", dest="outputPath", type=str,
                               help='base dir for outputs')
    module_parser.add_argument("-m", dest="model", type=str,
                               help='model')
    module_parser.add_argument("-w", dest="modelWeights", type=str,
                               default='imagenet',
                               help='model weights')
    module_parser.add_argument("-n", dest="note",
                               type=str,
                               default='',
                               help='note: will be added to output file path')
    return module_parser


def main_driver(XPath, outputPath,
                model, modelWeights,
                note):

    print('x path:', XPath)
    set_image_data_format('channels_last')
    assert(os.path.isfile(XPath))
    X = np.load(XPath)
    if not(os.path.isdir(outputPath)):
        os.makedirs(outputPath)
    modelFeatures = getModelFeatures(X,
                                     model,
                                     modelWeights)
    featurePath = join(outputPath, model)
    if not(os.path.isdir(featurePath)):
        os.makedirs(featurePath)
    np.save(join(featurePath, "{}_features_{}".format(model, note)),
            modelFeatures)
    with open(os.path.join(featurePath, 'dataInfo.txt'), 'w') as fid:
        fid.write("x path: {} \n".format(XPath))


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.XPath,
                    args.outputPath,
                    args.model,
                    args.modelWeights,
                    args.note)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()