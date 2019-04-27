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
from keras.callbacks import ModelCheckpoint
import datetime
import pickle


def getModel(modelName='InceptionV3',
             inputShape=(224,224,3),
             nClasses=4):
    """

    :param modelName:
    :param inputShape:
    :param nClasses:
    :return:
    """
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
    loss = 'categorical_crossentropy'
    metric = ['accuracy', 'categorical_crossentropy']
    model.compile(optimizer=adam, loss=loss, metrics=metric)
    return model


def trainModel(xTrn, yTrn,
               XVal, yVal,
               outputPath,
               modelName, d,
               nEpochs=100,
               batchSize=30):

    if d:
        nEpochs = 3
        nDebug = 200
        xTrn, yTrn = xTrn[0:nDebug], yTrn[0:nDebug]
        if not (XVal is None) and not (yVal is None):
            XVal, yVal = XVal[0:nDebug], yVal[0:nDebug]

    now = datetime.datetime.now()
    today = str(now.date())
    #set_image_data_format('channels_last')
    model = getModel(modelName=modelName)
    print(model.summary())
    modelOutputDir = os.path.join(outputPath,
                                   modelName + '_' +
                                   today)
    if not(os.path.isdir(modelOutputDir)):
        os.mkdir(modelOutputDir)
    yTrn = to_categorical(yTrn)
    if not(XVal is None) and not(yVal is None):
        yVal = to_categorical(yVal)
        valData = (XVal, yVal)
    else:
        valData = None

    # Set Callbacks
    modelOutPath = os.path.join(modelOutputDir, 'modelName.hdf5')
    modelCheckpoint = ModelCheckpoint(modelOutPath, monitor='val_loss',
                                      save_best_only=True,
                                      mode='auto', period=1)
    callbacks = [modelCheckpoint]
    history = model.fit(x=xTrn,
                        y=yTrn,
                        batch_size=batchSize,
                        epochs=nEpochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valData,
                        shuffle=True)
    historyPath = os.path.join(modelOutputDir, 'modelHistory.pickle')
    pickle.dump(history, open(historyPath, 'wb'))
    model.save(os.path.join(modelOutputDir, 'modelName_final.hdf5'))
    print('done!')


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-xtrn", dest="XTrainPath", type=str,
                               help="X train path ")
    module_parser.add_argument("-xval", dest="XValPath", type=str,
                               default='',
                               help="X val path")
    module_parser.add_argument("-ytrn", dest="yTrainPath", type=str,
                               help="y train path ")
    module_parser.add_argument("-yval", dest="yValPath", type=str,
                               default='',
                               help="y val path")
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


def main_driver(XTrainPath, yTrainPath,
                XValPath, yValPath,
                outputPath, model, d):
    d = bool(d)
    if d:
        print('debugging mode: ON')
    set_image_data_format('channels_last')
    assert(os.path.isfile(XTrainPath))
    assert(os.path.isfile(yTrainPath))
    xTrn = np.load(XTrainPath)
    yTrn = np.load(yTrainPath)
    if os.path.isfile(XValPath) and os.path.isfile(yValPath):
        XVal = np.load(XValPath)
        yVal = np.load(yValPath)
    else:
        XVal = None
        yVal = None
    if not(os.path.isdir(outputPath)):
        os.mkdir(outputPath)
    trainModel(xTrn, yTrn,
               XVal, yVal,
               outputPath,
               model, d)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.XTrainPath,
                    args.yTrainPath,
                    args.XValPath,
                    args.yValPath,
                    args.outputPath,
                    args.model,
                    args.d)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
