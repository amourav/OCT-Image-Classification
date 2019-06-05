from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback
import sys
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from keras.applications.vgg16 import preprocess_input as preprocess_input_xception
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, Flatten
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
from keras.backend import set_image_data_format
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import datetime
import time
import pickle


def getModel(modelName,
             inputShape,
             nClasses=4,
             lastLayer=1024,
             weights='imagenet'):
    """
    model getter
    :param modelName (str): Name of CNN model to train. Either [InceptionV3 or VGG16]
    :param inputShape (tuple-(Xres, YRes, nChannels)): Input shape for CNN model.
    :param nClasses (int): NUmber of unique output classes.
    :return: model: Keras CNN Model.
    """
    input_tensor = Input(shape=inputShape)
    #K.set_learning_phase(False)
    if modelName == 'InceptionV3':
        base_model = InceptionV3(weights=weights,
                                 include_top=False,
                                 input_tensor=input_tensor)
        #K.set_learning_phase(True)
        lastLayer = None
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif modelName == 'VGG16':
        base_model = VGG16(weights=weights,
                           include_top=False,
                           input_tensor=input_tensor)
        #K.set_learning_phase(True)
        #lastLayer = 4096
        x = base_model.output
        x = Flatten()(x)
        x = Dense(lastLayer, activation='relu')(x)
        x = Dense(lastLayer, activation='relu')(x)
    elif modelName == 'ResNet50':
        base_model = ResNet50(weights=weights,
                              include_top=False,
                              input_tensor=input_tensor)
        #K.set_learning_phase(True)
        #lastLayer = 2048
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #x = Dense(lastLayer, activation='relu')(x)
    elif modelName == 'Xception':
        base_model = Xception(weights=weights,
                              include_top=False,
                              input_tensor=input_tensor)
        #K.set_learning_phase(True)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    else:
        raise Exception('model name not recognized')
    # add a global spatial average pooling layer

    """
    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(lastLayer, activation='relu')(x)
    x = Dense(lastLayer, activation='relu')(x)
    # and a logistic layer
    """
    predictions = Dense(nClasses, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    #for layer in base_model.layers:
    #    layer.trainable = False
    for layer in base_model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            print('BN layer: trainable')
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False
    adam = Adam(lr=0.0001)
    # compile the model (should be done *after* setting layers to non-trainable)
    loss = 'categorical_crossentropy'
    metric = ['accuracy', 'categorical_crossentropy']
    model.compile(optimizer=adam, loss=loss, metrics=metric)
    return model


def getPreprocess(modelName):
    if modelName == 'InceptionV3':
        preprocessInput = preprocess_input_inception_v3
    elif modelName == 'VGG16':
        #preprocessInput = preprocess_input_vgg16
        preprocessInput = None
    elif modelName == 'ResNet50':
        preprocessInput = preprocess_input_ResNet50
    elif modelName == 'Xception':
        preprocessInput = preprocess_input_xception
    else:
        raise Exception('model name not recognized')
    return preprocessInput


def trainModel(xTrn, yTrn,
               XVal, yVal,
               outputPath,
               modelName,
               modelWeights,
               aug, d, note,
               xTest = None,
               nEpochs=250,
               batchSize=30,
               lastLayer=4096):
    """
    Train CNN Model
    :param xTrn (npy array): Input training image data (nTrain, Xres, Yres, nChannels=3)
    :param yTrn (npy array): Target training image classes (nTrain, )
    :param XVal (npy array): Input val image data (nTrain, Xres, Yres, nChannels=3)
    :param yVal (npy array): Target val image classes (nTrain, )
    :param outputPath (str): Path to output dir.
    :param modelName (str): Name of pretrained Keras model.
    :param d (bool):
    :param nEpochs (int): Number of epochs to train model.
    :param batchSize (int): Number of images to use in each training batch.
    :return: None
    """
    print(modelName)
    preprocessInput = getPreprocess(modelName)
    if d:
        nEpochs = 3
        batchSize = 2
        nDebug = 20
        xTrn, yTrn = xTrn[0:nDebug], yTrn[0:nDebug]
        if not (XVal is None) and not (yVal is None):
            XVal, yVal = XVal[0:nDebug], yVal[0:nDebug]
        if xTest is not None:
            xTest = xTest[0:nDebug]

    now = datetime.datetime.now()
    today = str(now.date()) + \
                '_' + str(now.hour) + \
                '_' + str(now.minute)
    xShape = xTrn.shape
    xShape = (xShape[1], xShape[2], xShape[3])
    model = getModel(modelName=modelName, 
                     inputShape=xShape,
                     weights=modelWeights,
                     lastLayer=lastLayer)
    print(note)
    print(model.summary())
    modelOutputDir = os.path.join(outputPath,
                                   modelName + '_' +
                                   "dataAug_" + str(aug) +
                                   today + '_' + note)
    if not(os.path.isdir(modelOutputDir)):
        os.mkdir(modelOutputDir)
    if preprocessInput is not None:
        xTrn = preprocessInput(xTrn)
    yTrn = to_categorical(yTrn)
    if not(XVal is None) and not(yVal is None):
        if preprocessInput is not None:
            XVal = preprocessInput(XVal)
        yVal = to_categorical(yVal)
        valData = (XVal, yVal)
    else:
        valData = None

    # Set Callbacks
    modelOutPath = os.path.join(modelOutputDir, '{}.hdf5'.format(modelName))
    modelCheckpoint = ModelCheckpoint(modelOutPath, monitor='val_loss',
                                      save_best_only=True,
                                      mode='auto', period=1)
    callbacks = [modelCheckpoint]
    t0 = time.time()
    if not(aug):
        history = model.fit(x=xTrn,
                            y=yTrn,
                            batch_size=batchSize,
                            epochs=nEpochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=valData,
                            shuffle=True)
    else:
        print('fitting image generator')
        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0,
            zoom_range=0.1,  
            brightness_range=(0.90, 1.1),
            shear_range=15,
            fill_mode='nearest',
            vertical_flip=False,
            horizontal_flip=True)
        history = model.fit_generator(datagen.flow(xTrn, yTrn,
                                                   batch_size=batchSize),
                                      steps_per_epoch=len(xTrn) / batchSize,
                                      epochs=nEpochs, callbacks=callbacks,
                                      validation_data=valData, shuffle=True,
                                      verbose=1)
    dt = (time.time() - t0)/3600
    print("training time: {} h".format(dt))
    historyPath = os.path.join(modelOutputDir, '{}_History.csv'.format(modelName))
    hist = history.history
    histDf = pd.DataFrame(hist)
    histDf.to_csv(historyPath)

    # Run inference on test set if provided
    if xTest is not None:
        if preprocessInput is not None:
            xTest = preprocessInput(xTest)
        print('running model pred on test set')
        yTestPred = model.predict(xTest,
                                  batch_size=20,
                                  verbose=1)
        yTestPredPath = os.path.join(modelOutputDir, 'yTestPred.npy')
        np.save(yTestPredPath, yTestPred)

    model.save(os.path.join(modelOutputDir, '{}_final.hdf5'.format(modelName)))
    with open(os.path.join(modelOutputDir, 'trnInfo.txt'), 'w') as fid:
        fid.write("model: {} \n".format(modelName))
        fid.write("x shape: {} \n".format(xShape))
        fid.write("nEpochs: {} \n".format(nEpochs))
        fid.write("batchSize: {} \n".format(batchSize))
        fid.write("lastLayer: {} \n".format(lastLayer))
    return modelOutputDir


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
    module_parser.add_argument("-xtest", dest="XTestPath", type=str,
                               default=None,
                               help='model weights')
    module_parser.add_argument("-o", dest="outputPath", type=str,
                               help='base dir for outputs')
    module_parser.add_argument("-m", dest="model", type=str,
                               default='InceptionV3',
                               help='model (default: inception)')
    module_parser.add_argument("-w", dest="modelWeights", type=str,
                               default='imagenet',
                               help='model weights')
    module_parser.add_argument("-aug", dest="aug",
                               type=int,
                               default=0,
                               choices=[1, 0],
                               help='augment: 1 - ON, 0 - OFF')
    module_parser.add_argument("-d", dest="d",
                               type=int,
                               default=0,
                               choices=[1, 0],
                               help='debug: 1 - ON, 0 - OFF')
    module_parser.add_argument("-n", dest="note",
                               type=str,
                               default='',
                               help='note: will be added to output file path')
    return module_parser


def main_driver(XTrainPath, yTrainPath,
                XValPath, yValPath,
                XTestPath,
                outputPath, model,
                modelWeights,
                aug, d, note):
    """
    Load Training and Validation data and call trainModel.
    :param XTrainPath (str): path to training image data
    :param yTrainPath (str): path to training target classes
    :param XValPath (str): path to val image data
    :param yValPath (str): path to val target classes
    :param outputPath (str): path to output dir
    :param model (str): Name of Keras pretrained model
    :param d (str): set d=1 to debug.
    :return: None
    """
    print('trn path:', XTrainPath)
    aug = bool(aug)
    print("data augmentation: {}".format(aug))
    d = bool(d)
    if d:
        print('debugging mode: ON')
    set_image_data_format('channels_last')
    assert(os.path.isfile(XTrainPath))
    assert(os.path.isfile(yTrainPath))
    if XTestPath is not None:
        assert(os.path.isfile(XTestPath))
        xTest = np.load(XTestPath)
    xTrn = np.load(XTrainPath)
    yTrn = np.load(yTrainPath)


    if os.path.isfile(XValPath) and os.path.isfile(yValPath):
        XVal = np.load(XValPath)
        yVal = np.load(yValPath)
    else:
        XVal = None
        yVal = None
    if not(os.path.isdir(outputPath)):
        os.makedirs(outputPath)
    modelOutputDir = trainModel(xTrn, yTrn,
                     XVal, yVal,
                     outputPath,
                     model, modelWeights,
                     aug, d, note, xTest=xTest)
    with open(os.path.join(modelOutputDir, 'dataInfo.txt'), 'w') as fid:
        fid.write("XTrainPath: {} \n".format(XTrainPath))
        fid.write("XValPath: {} \n".format(XValPath))
        fid.write("XTestPath: {} \n".format(str(XTestPath)))


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.XTrainPath,
                    args.yTrainPath,
                    args.XValPath,
                    args.yValPath,
                    args.XTestPath,
                    args.outputPath,
                    args.model,
                    args.modelWeights,
                    args.aug,
                    args.d,
                    args.note)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
