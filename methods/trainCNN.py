from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import traceback
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
from numpy.random import seed
import datetime
import time
import skimage


def getModel(modelName,
             inputShape,
             nClasses=4,
             weights='imagenet'):
    """
    get keras model
    :param modelName: Name of CNN model to train (str)
    Either [InceptionV3, VGG16, Xception, or ResNet50]
    :param inputShape: Input shape for CNN model [Xres, YRes, nChannels] (tuple)
    :param nClasses:  NUmber of unique output classes (int)
    :param weights: path to pretrained model weights or enter
    string 'imagenet' to automatically download weights (str)
    :return: keras model
    """

    input_tensor = Input(shape=inputShape)
    if modelName == 'InceptionV3':
        base_model = InceptionV3(weights=weights,
                                 include_top=False,
                                 input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif modelName == 'VGG16':
        lastLayer = 4096  # number of nodes
        base_model = VGG16(weights=weights,
                           include_top=False,
                           input_tensor=input_tensor)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(lastLayer, activation='relu')(x)
        x = Dense(lastLayer, activation='relu')(x)
    elif modelName == 'ResNet50':
        base_model = ResNet50(weights=weights,
                              include_top=False,
                              input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif modelName == 'Xception':
        base_model = Xception(weights=weights,
                              include_top=False,
                              input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    else:
        raise Exception('model name not recognized')

    predictions = Dense(nClasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False
    adam = Adam(lr=0.0001)
    loss = 'categorical_crossentropy'
    metric = ['accuracy', 'categorical_crossentropy']
    model.compile(optimizer=adam, loss=loss, metrics=metric)
    return model


def preprocessInputVGG16(X, newSize=(224, 224, 3)):
    """
    preprocess image data for VGG16 by resizing
    :param X: image data array [nSamples, xRes, yRes, channels] (np array)
    :param newSize: new resolution of each sample [xResNew, yResNew, nChannels] (tuple)
    :return: resized image array stack [nSamples, xResNew, yResNew, nChannels] (np array)
    """
    xShape = X.shape
    if not((xShape[1], xShape[2]) == (newSize[0], newSize[1])):
        xResized = []
        for xi in X:
            xiR = skimage.transform.resize(xi, newSize)
            xResized.append(xiR)
        xResized = np.stack(xResized, axis=0)
        return xResized
    else:
        return X


def getPreprocess(modelName):
    """
    retrieve the model specific preprocessing function
    :param modelName: name of pretrained model (str)
    :return: preprocessing function
    """
    if modelName == 'InceptionV3':
        preprocessInput = preprocess_input_inception_v3
    elif modelName == 'VGG16':
        preprocessInput = preprocessInputVGG16
    elif modelName == 'ResNet50':
        preprocessInput = preprocess_input_ResNet50
    elif modelName == 'Xception':
        preprocessInput = preprocess_input_xception
    else:
        raise Exception('model name not recognized')
    return preprocessInput


def trainModel(xTrn, yTrn,
               XVal, yVal,
               modelOutputDir,
               modelName,
               modelWeights,
               aug, d, note,
               xTest = None,
               nEpochs=300,
               batchSize=30):
    """
    train CNN
    :param xTrn: training data image stack [n_train, xRes, yRes, channels=3] (npy array)
    :param yTrn: training image labels [nSamples] (npy array)
    :param XVal: validation data image stack [n_train, xRes, yRes, channels=3] (npy array)
    If set to None, validation data will be selected from xTrn
    :param yVal: If set to None, validation data will be selected from yTrn
    :param outputPath: output path for training output (str)
    :param modelName: Name of CNN model to train (str)
    one of - [InceptionV3, VGG16, Xception, or ResNet50]
    :param modelWeights: path to pretrained model weights (str)
    or set to imagenet to download the pretrained model weights
    :param aug: enable data augmentation (1=On, 0=Off) (int/bool)
    :param d: debugging mode [limit dataset size and training iterations] (int/bool)
    1=On, 0=Off
    :param note: print during model training (str)
    :param xTest: test data image stack [n_train, xRes, yRes, channels=3] (npy array)
    [optional]
    :param nEpochs: number of training epochs (int)
    :param batchSize: batch size (int)
    :return: modelOutputDir: path to model output directory (str)
    """
    # set random seed for numpy and tensorflow
    seed(0)
    set_random_seed(0)
    set_image_data_format('channels_last')
    print(modelName, note)

    # reduce dataset, epochs, and batch size for debugging mode
    if d:
        nEpochs = 2
        batchSize = 2
        nDebug = 6
        xTrn, yTrn = xTrn[0:nDebug], yTrn[0:nDebug]
        if not (XVal is None) and not (yVal is None):
            XVal, yVal = XVal[0:nDebug], yVal[0:nDebug]
        if xTest is not None:
            xTest = xTest[0:nDebug]

    xShape = xTrn.shape
    xShape = (xShape[1], xShape[2], xShape[3])
    model = getModel(modelName=modelName, 
                     inputShape=xShape,
                     weights=modelWeights)
    print(model.summary())
    # normalize data to the network's specifications
    preprocessInput = getPreprocess(modelName)
    xTrn = preprocessInput(xTrn)
    yTrn = to_categorical(yTrn)

    # Set Validation Data
    valSplit = 0.0
    if not(XVal is None) and not(yVal is None):
        XVal = preprocessInput(XVal)
        yVal = to_categorical(yVal)
        valData = (XVal, yVal)
    else:
        valData = None
        valSplit = 0.1

    # Set Callbacks
    modelOutPath = os.path.join(modelOutputDir, '{}.hdf5'.format(modelName))
    modelCheckpoint = ModelCheckpoint(modelOutPath, monitor='val_loss',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='auto', period=1)
    earlyStop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=150,
                              verbose=1,
                              restore_best_weights=True)
    callbacks = [modelCheckpoint, earlyStop]

    # train model
    t0 = time.time()
    if not aug:
        history = model.fit(x=xTrn,
                            y=yTrn,
                            batch_size=batchSize,
                            epochs=nEpochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=valData,
                            validation_split=valSplit,
                            shuffle=True)
    else:
        print('fitting image generator')
        datagen = ImageDataGenerator(featurewise_center=False,
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

    # save history output
    historyPath = os.path.join(modelOutputDir, '{}_History.csv'.format(modelName))
    hist = history.history
    histDf = pd.DataFrame(hist)
    histDf.to_csv(historyPath)

    # save model architecture to json
    jsonPath = os.path.join(modelOutputDir, '{}_architecture.json'.format(modelName))
    modelJson = model.to_json()
    with open(jsonPath, "w") as json_file:
        json_file.write(modelJson)

    # Run inference on test set if provided
    if xTest is not None:
        xTest = preprocessInput(xTest)
        print('running model pred on test set')
        yTestPred = model.predict(xTest,
                                  batch_size=20,
                                  verbose=1)
        yTestPredPath = os.path.join(modelOutputDir, 'yTestPred.npy')
        np.save(yTestPredPath, yTestPred)

    with open(os.path.join(modelOutputDir, 'trnInfo.txt'), 'w') as fid:
        fid.write("model: {} \n".format(modelName))
        fid.write("x shape: {} \n".format(xShape))
        fid.write("nEpochs: {} \n".format(nEpochs))
        fid.write("batchSize: {} \n".format(batchSize))


def loadTargetData(yPath):
    """
    load target data labels
    :param yPath: path to image labels file (str)
    :return: numpy assay of image labels (npy arr), index for images ()
    """
    if yPath.endswith('.npy'):
        yArr = np.load(yPath)
        idx = np.arange(len(yArr))
    elif yPath.endswith('.csv'):
        yArr = pd.read_csv(yPath,
                           index_col=0)
        idx = yArr.index
        yArr = yArr.values
    else:
        raise Exception('unknown file type: {}'.format(yPath))
    return yArr, idx


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-xtrn", dest="XTrainPath", type=str,
                               help="X train path")
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
    :param XTrainPath (str): path to preprocessed training image data
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
    assert(os.path.isfile(XTrainPath))
    assert(os.path.isfile(yTrainPath))
    if XTestPath is not None:
        assert(os.path.isfile(XTestPath))
        xTest = np.load(XTestPath)
    """############################################################################
                        0. Load Data
    ############################################################################"""
    xTrn = np.load(XTrainPath)
    yTrn, idx = loadTargetData(yTrainPath)

    if os.path.isfile(XValPath) and os.path.isfile(yValPath):
        XVal = np.load(XValPath)
        yVal, idx = loadTargetData(yValPath)
    else:
        XVal = None
        yVal = None

    print(note)
    now = datetime.datetime.now()
    today = str(now.date()) + \
                '_' + str(now.hour) + \
                '_' + str(now.minute)
    modelOutputDir = os.path.join(outputPath,
                                  model + '_' +
                                  today + '_' + note)
    if not(os.path.isdir(modelOutputDir)):
        os.makedirs(modelOutputDir)
    """############################################################################
                        1. train CNN and save training info
    ############################################################################"""
    trainModel(xTrn, yTrn,
               XVal, yVal,
               outputPath,
               model,
               modelWeights,
               aug, d,
               note,
               xTest=xTest)

    #save training info
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
