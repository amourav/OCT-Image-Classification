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


def get_model(model_name,
              input_shape,
              n_classes=4,
              weights='imagenet'):
    """
    get keras model
    :param model_name: Name of CNN model to train (str)
    Either [InceptionV3, VGG16, Xception, or ResNet50]
    :param input_shape: Input shape for CNN model [Xres, YRes, nChannels] (tuple)
    :param n_classes:  NUmber of unique output classes (int)
    :param weights: path to pretrained model weights or enter
    string 'imagenet' to automatically download weights (str)
    :return: keras model
    """
    input_tensor = Input(shape=input_shape)
    if model_name == 'InceptionV3':
        base_model = InceptionV3(weights=weights,
                                 include_top=False,
                                 input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif model_name == 'VGG16':
        last_layer = 4096  # number of nodes
        base_model = VGG16(weights=weights,
                           include_top=False,
                           input_tensor=input_tensor)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(last_layer, activation='relu')(x)
        x = Dense(last_layer, activation='relu')(x)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights=weights,
                              include_top=False,
                              input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif model_name == 'Xception':
        base_model = Xception(weights=weights,
                              include_top=False,
                              input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    else:
        raise Exception('model name not recognized')

    predictions = Dense(n_classes, activation='softmax')(x)
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


def preprocess_input_vgg16(x, new_size=(224, 224, 3)):
    """
    preprocess image data for VGG16 by resizing
    :param x: image data array [nSamples, x_res, y_res, channels] (np array)
    :param new_size: new resolution of each sample [xResNew, yResNew, nChannels] (tuple)
    :return: resized image array stack [nSamples, xResNew, yResNew, nChannels] (np array)
    """
    if (x.shape[1], x.shape[2]) == (new_size[0], new_size[1]):
        return x

    x_resized = []
    for xi in x:
        xi_r = skimage.transform.resize(xi, new_size)
        x_resized.append(xi_r)
    x_resized = np.stack(x_resized, axis=0)
    return x_resized


def get_preprocess(model_name):
    """
    retrieve the model specific preprocessing function
    :param model_name: name of pretrained model (str)
    :return: preprocessing function
    """
    if model_name == 'InceptionV3':
        preprocess_input = preprocess_input_inception_v3
    elif model_name == 'VGG16':
        preprocess_input = preprocess_input_vgg16
    elif model_name == 'ResNet50':
        preprocess_input = preprocess_input_ResNet50
    elif model_name == 'Xception':
        preprocess_input = preprocess_input_xception
    else:
        raise Exception('model name not recognized')
    return preprocess_input


def get_debug_params(x_trn, y_trn, x_val, y_val, x_test,
                     n_epochs=2, batch_size=2, n_debug=6):
    x_trn, y_trn = x_trn[0:n_debug], y_trn[0:n_debug]
    if not (x_val is None) and not (y_val is None):
        x_val, y_val = x_val[0:n_debug], y_val[0:n_debug]
    if x_test is not None:
        x_test = x_test[0:n_debug]
    debug_params = (x_trn, y_trn, x_val, y_val,
                    x_test, n_epochs, batch_size
                    )
    return debug_params


def set_val_data(preprocess_input, x_val, y_val):
    val_split = 0.0
    if not (x_val is None) and not (y_val is None):
        x_val = preprocess_input(x_val)
        y_val = to_categorical(y_val)
        val_data = (x_val, y_val)
    else:
        val_data = None
        val_split = 0.1
    return val_data, val_split


def set_callbacks(model_out_path):
    model_checkpoint = ModelCheckpoint(model_out_path, monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto', period=1)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=150,
                               verbose=1,
                               restore_best_weights=True)
    callbacks = [model_checkpoint, early_stop]
    return callbacks


def save_model(model_output_dir, model_name, history, model):
    history_path = os.path.join(model_output_dir, '{}_History.csv'.format(model_name))
    hist = history.history
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(history_path)

    # save model architecture to json
    json_path = os.path.join(model_output_dir, '{}_architecture.json'.format(model_name))
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)


def model_pred(x_test, preprocess_input, model,
               model_output_dir, model_name,
               x_shape, n_epochs, batch_size):
    x_test = preprocess_input(x_test)
    print('running model pred on test set')
    y_test_pred = model.predict(x_test,
                                batch_size=20,
                                verbose=1)
    y_test_pred_path = os.path.join(model_output_dir, 'yTestPred.npy')
    np.save(y_test_pred_path, y_test_pred)

    with open(os.path.join(model_output_dir, 'trnInfo.txt'), 'w') as fid:
        fid.write("model: {} \n".format(model_name))
        fid.write("x shape: {} \n".format(x_shape))
        fid.write("n_epochs: {} \n".format(n_epochs))
        fid.write("batch_size: {} \n".format(batch_size))


def train_model(x_trn, y_trn,
                x_val, y_val,
                model_output_dir,
                model_name,
                model_weights,
                aug, d, note,
                x_test=None,
                n_epochs=300,
                batch_size=30):
    """
    train CNN
    :param x_trn: training data image stack [n_train, x_res, y_res, channels=3] (npy array)
    :param y_trn: training image labels [nSamples] (npy array)
    :param x_val: validation data image stack [n_train, x_res, y_res, channels=3] (npy array)
    If set to None, validation data will be selected from x_trn
    :param y_val: If set to None, validation data will be selected from y_trn
    :param output_path: output path for training output (str)
    :param model_name: Name of CNN model to train (str)
    one of - [InceptionV3, VGG16, Xception, or ResNet50]
    :param model_weights: path to pretrained model weights (str)
    or set to imagenet to download the pretrained model weights
    :param aug: enable data augmentation (1=On, 0=Off) (int/bool)
    :param d: debugging mode [limit dataset size and training iterations] (int/bool)
    1=On, 0=Off
    :param note: print during model training (str)
    :param x_test: test data image stack [n_train, x_res, y_res, channels=3] (npy array)
    [optional]
    :param n_epochs: number of training epochs (int)
    :param batch_size: batch size (int)
    :return: model_output_dir: path to model output directory (str)
    """
    # set random seed for numpy and tensorflow
    seed(0)
    set_random_seed(0)
    set_image_data_format('channels_last')
    print(model_name, note)

    # reduce dataset, epochs, and batch size for debugging mode
    if d:
        debug_params = get_debug_params(x_trn, y_trn, x_val, y_val, x_test,
                                        n_epochs=2, batch_size=2, n_debug=6)
        x_trn, y_trn, x_val, y_val = debug_params[:4]
        x_test, n_epochs, batch_size = debug_params[4:]

    x_shape = x_trn.shape[1:4]
    model = get_model(model_name=model_name,
                      input_shape=x_shape,
                      weights=model_weights)
    print(model.summary())
    # normalize data to the network's specifications
    preprocess_input = get_preprocess(model_name)
    x_trn = preprocess_input(x_trn)
    y_trn = to_categorical(y_trn)

    # Set Validation Data
    val_data, val_split = set_val_data(preprocess_input, x_val, y_val)

    # Set Callbacks
    model_out_path = os.path.join(model_output_dir, '{}.hdf5'.format(model_name))
    callbacks = set_callbacks(model_out_path)

    # train model
    t0 = time.time()
    if not aug:
        history = model.fit(x=x_trn,
                            y=y_trn,
                            batch_size=batch_size,
                            epochs=n_epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=val_data,
                            validation_split=val_split,
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

        history = model.fit_generator(datagen.flow(x_trn, y_trn,
                                                   batch_size=batch_size),
                                      steps_per_epoch=len(x_trn) / batch_size,
                                      epochs=n_epochs, callbacks=callbacks,
                                      validation_data=val_data, shuffle=True,
                                      verbose=1)
    dt = (time.time() - t0) / 3600
    print("training time: {} h".format(dt))
    # save history output
    save_model(model_output_dir, model_name, history, model)

    # Run inference on test set if provided
    if x_test is not None:
        model_pred(x_test, preprocess_input, model,
                   model_output_dir, model_name,
                   x_shape, n_epochs, batch_size)


def load_target_data(y_path):
    """
    load target data labels
    :param y_path: path to image labels file (str)
    :return: numpy assay of image labels (npy arr), index for images ()
    """
    if y_path.endswith('.npy'):
        y_arr = np.load(y_path)
        idx = np.arange(len(y_arr))
    elif y_path.endswith('.csv'):
        y_arr = pd.read_csv(y_path,
                            index_col=0)
        idx = y_arr.index
        y_arr = y_arr.values
    else:
        raise Exception('unknown file type: {}'.format(y_path))
    return y_arr, idx


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-xtrn", dest="x_train_path", type=str,
                               help="x train path")
    module_parser.add_argument("-xval", dest="x_val_path", type=str,
                               default='',
                               help="x val path")
    module_parser.add_argument("-ytrn", dest="y_train_path", type=str,
                               help="y train path ")
    module_parser.add_argument("-yval", dest="y_val_path", type=str,
                               default='',
                               help="y val path")
    module_parser.add_argument("-xtest", dest="x_test_path", type=str,
                               default=None,
                               help='model weights')
    module_parser.add_argument("-o", dest="output_path", type=str,
                               help='base dir for outputs')
    module_parser.add_argument("-m", dest="model", type=str,
                               default='InceptionV3',
                               help='model (default: inception)')
    module_parser.add_argument("-w", dest="model_weights", type=str,
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


def main_driver(x_train_path, y_train_path,
                x_val_path, y_val_path,
                x_test_path,
                output_path, model,
                model_weights,
                aug, d, note):
    """
    Load Training and Validation data and call train_model.
    :param x_train_path (str): path to preprocessed training image data
    :param y_train_path (str): path to training target classes
    :param x_val_path (str): path to val image data
    :param y_val_path (str): path to val target classes
    :param output_path (str): path to output dir
    :param model (str): Name of Keras pretrained model
    :param d (str): set d=1 to debug.
    :return: None
    """

    print('trn path:', x_train_path)
    aug = bool(aug)
    print("data augmentation: {}".format(aug))
    d = bool(d)
    if d:
        print('debugging mode: ON')
    assert (os.path.isfile(x_train_path))
    assert (os.path.isfile(y_train_path))
    if x_test_path is not None:
        assert (os.path.isfile(x_test_path))
        x_test = np.load(x_test_path)
    """############################################################################
                        0. Load Data
    ############################################################################"""
    x_trn = np.load(x_train_path)
    y_trn, idx = load_target_data(y_train_path)

    if os.path.isfile(x_val_path) and os.path.isfile(y_val_path):
        x_val = np.load(x_val_path)
        y_val, idx = load_target_data(y_val_path)
    else:
        x_val = None
        y_val = None

    print(note)
    now = datetime.datetime.now()
    today = str(now.date()) + \
            '_' + str(now.hour) + \
            '_' + str(now.minute)
    model_output_dir = os.path.join(output_path,
                                    model + '_' +
                                    today + '_' + note)
    if not (os.path.isdir(model_output_dir)):
        os.makedirs(model_output_dir)
    """############################################################################
                        1. train CNN and save training info
    ############################################################################"""
    train_model(x_trn, y_trn,
                x_val, y_val,
                output_path,
                model,
                model_weights,
                aug, d,
                note,
                x_test=x_test)

    # save training info
    with open(os.path.join(model_output_dir, 'dataInfo.txt'), 'w') as fid:
        fid.write("x_train_path: {} \n".format(x_train_path))
        fid.write("x_val_path: {} \n".format(x_val_path))
        fid.write("x_test_path: {} \n".format(str(x_test_path)))


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.x_train_path,
                    args.y_train_path,
                    args.x_val_path,
                    args.y_val_path,
                    args.x_test_path,
                    args.output_path,
                    args.model,
                    args.model_weights,
                    args.aug,
                    args.d,
                    args.note)
        print('Done!')

    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
