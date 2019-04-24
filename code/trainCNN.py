from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
#from preprocessingUtils import normImg
import traceback
import sys
from PIL import Image
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.backend import set_image_data_format
import datetime

def simpleANN():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def getModel(modelName='InceptionV3',
             inputShape=(224,224,3),
             nClasses=4):
    input_tensor = Input(shape=inputShape)
    if modelName == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_tensor=input_tensor)
    elif modelName == 'VGG16':
        base_model = VGG16(weights='imagenet',
                           include_top=False,
                           input_tensor=input_tensor)
    else:
        raise Exception('model name not recognized')
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nClasses, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    adam = Adam(lr=0.0001)
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def getCNN(modelName='VGG16', inputShape=(224,224,3)):
    nClasses = 4
    #inputs = Input(inputShape)
    if modelName == 'inception':
        print('loading inceptionV3')
        baseLayers = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_shape=inputShape)
    elif modelName == "VGG16":
        print('loading VGG16')
        baseLayers = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_shape=inputShape)
    else:
        raise Exception('model name not recognized')
    x = baseLayers.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    yPred = Dense(nClasses, activation='softmax')(x)
    model = Model(input=baseLayers.inputs, output=yPred)

    for layer in baseLayers.layers:
        layer.trainable = False
    print('compiling model')
    adam_ = Adam(lr=0.0001)
    model.compile(model, loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='rmsprop')
    return model


def trainModel(trainPath, valPath, outputPath, modelName, debug_):
    now = datetime.datetime.now()
    today = str(now.date())
    set_image_data_format('channels_last')
    model = getModel(modelName=modelName)

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
                               default='InceptionV3',
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
