from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
#from preprocessingUtils import normImg
import traceback
import sys
from PIL import Image
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.backend import set_image_data_format
import datetime

def getCNN(modelName, inputShape=(224,224,1)):
    nClasses = 4
    inputs = Input(inputShape)
    if modelName == 'inception':
        baseLayers = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_tensor=inputs)
    x = baseLayers.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    yPred = Dense(nClasses, activation='softmax')(x)
    model = Model(input=inputs, output=yPred)

    for layer in baseLayers.layers:
        layer.trainable = False
    print('compiling model')
    adam_ = Adam(lr=0.0001)
    model.compile(model, metrics=['accuracy'], optimizer=adam_)
    return model


def trainModel(trainPath, valPath, outputPath, model, debug_):
    now = datetime.datetime.now()
    today = str(now.date())
    pass

def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i1", dest="trainPath", type=str,
                               help="the location dataset")
    module_parser.add_argument("-i2", dest="valPath", type=str,
                               default='',
                               help="the location dataset")
    module_parser.add_argument("-o", dest="outputPath", type=str,
                               help='base dir for outputs')
    module_parser.add_argument("-m", dest="model", type=str,
                               default='inception',
                               help='model (default: inception)')
    module_parser.add_argument("-d", dest="d",
                               type=int,
                               default=0,
                               choices=[1, 0],
                               help='debug: 1 - ON, 0 - OFF')
    return module_parser


def main_driver(trainPath, valPath, outputPath, model, d):
    d = bool(d)
    if d:
        print('debugging mode: ON')
    set_image_data_format('channels_last')
    assert(os.path.isfile(trainPath))
    assert(os.path.isfile(valPath))
    if not(os.path.isdir(outputPath)):
        os.mkdir(outputPath)
    trainModel(trainPath, valPath, outputPath, model, d)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.trainPath,
                    args.valPath,
                    args.outputPath,
                    args.model,
                    args.d)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
