from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import traceback
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.backend import set_image_data_format
from keras.models import model_from_json
import skimage
from PIL import Image
sys.path.append('./methods')
from trainCNN import get_preprocess
from preprocess import get_class_labels


def get_binary_pred(model_pred_df):
    """
    Get probability of Urgent vs Non-Urgent diagnosis from predicted class label probabilities
    :param model_pred_df: Dataframe containing multi-class probabilities for each sample
    :return: Dataframe containing binary probabilities for each sample
    """
    urgent_labels = ['CNV', 'DME']
    urgent_cols = []
    pred_urgent_df = pd.DataFrame(index=model_pred_df.index)
    for col in model_pred_df.columns:
        if (urgent_labels[0] in col) or (urgent_labels[1] in col):
            urgent_cols.append(col)
    assert (len(urgent_cols) == 2)
    pred_urgent_df['urgent_proba'] = model_pred_df[urgent_cols[0]] + model_pred_df[urgent_cols[1]]
    return pred_urgent_df


def mean_prediction(model_pred_df, y_vals=[0, 1, 2, 3]):
    """
    output the mean probability predicted for each class
    :param model_pred_df: dataframe containing the predictions
    from all models for each class (pandas Dataframe)
    :param y_vals: possible output classes (list like object)
    :return: mean probability for each class (pandas dataframe)
    """
    img_type_dict = get_class_labels(intKey=True)
    mean_pred = pd.DataFrame(index=model_pred_df.index)
    for yi in np.unique(y_vals):
        mean = model_pred_df.filter(regex='_{}'.format(yi)).mean(axis=1)
        mean_pred['proba_{}'.format(img_type_dict[yi])] = mean
    mean_pred = mean_pred.div(mean_pred.sum(axis=1), axis=0)
    return mean_pred


def load_model(model_name, ensemble_path):
    """
    load keras model from json and model weights
    :param model_name: name of model (str)
    :param ensemble_path: path of directory containing
    all models of the ensemble classifier (str)
    :return: keras model, directory containing model files
    """
    model_dirs = os.listdir(ensemble_path)
    model_dirs = [d for d in model_dirs if os.path.isdir(join(ensemble_path, d))]
    # find path to individual model
    model_dir = None
    for dir in model_dirs:
        if model_name in dir:
            model_dir = dir

    model_dir_path = join(ensemble_path,
                          model_dir)
    model_weights_path = join(model_dir_path,
                              model_name + ".hdf5")
    # load json and create model
    print('loading {} json'.format(model_name))
    json_path = join(os.path.dirname(model_weights_path),
                     '{}_architecture.json'.format(model_name))
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(model_weights_path)
    print("Loaded {} json from disk".format(model_name))
    return model, model_dir_path


def preprocess_imgs(x_path, new_size):
    """
    preprocess images in the directory and return data
    :param x_path: path to image directory containing .jpeg files (str)
    :param new_size: desired shape of each image array (tuple - [x_res, y_res, channels])
    :return: preprocessed image array (np array) [nImages, x_res, y_res, channels],
             list of image names (list)
    """
    img_files = os.listdir(x_path)
    img_files = [f for f in img_files if f.endswith('.jpeg')]
    img_stack = []
    for img_fname in img_files:
        img_path = os.path.join(x_path, img_fname)
        img_arr = np.array(Image.open(img_path))
        img_arr = skimage.transform.resize(img_arr, new_size)
        img_arr = (img_arr - img_arr.min()) / img_arr.max()
        img_stack.append(img_arr)
    img_stack = np.stack(img_stack, axis=0)
    img_names = [n.split('.')[0] for n in img_files]  # include text before .jpeg file ending
    return img_stack, img_names


def save_model_results(model_name, y_pred, img_names, model_dir_path, img_type_dict):
    """
    save predictions of individual models
    :param model_name: name of model (e.g. VGG16) (str)
    :param y_pred: model predictions (npy array)
    :param img_names: list of image filenames (list)
    :param model_dir_path: path to the directory containing the model (str)
    :param img_type_dict: mapping from integer labels to image labels (dict)
    :return: dataframe containing model predictions (pandas DF)
    """
    cols = ["{}_{}_{}".format(model_name, i, l)
            for (l, i)
            in img_type_dict.items()]

    y_pred_df = pd.DataFrame(y_pred,
                             columns=cols,
                             index=img_names)
    y_pred_df.to_csv(join(model_dir_path,
                          "{}_predictions.csv".format(model_name)))
    return y_pred_df


def save_predictions(model_pred_dict, models, img_names, out_path):
    """
    save the predictions of the ensemble classifier
    :param model_pred_dict: dictionary containing the predictions of each model (dict)
    :param models: list of model names (list)
    :param img_names: list of image filenames (list)
    :param out_path: directory where predictions are saved (str)
    :return:
    """
    # merge predictions into a single dataframe
    model_pred_df = pd.DataFrame(index=img_names)
    for modelName in models:
        model_pred_df = pd.merge(model_pred_df,
                                 model_pred_dict[modelName],
                                 left_index=True,
                                 right_index=True)

    # calculate average probability for each class
    mean_pred_df = mean_prediction(model_pred_df)
    binary_pred_df = get_binary_pred(mean_pred_df)

    # save dataframes to csv
    model_pred_df.to_csv(join(out_path,
                              "individualModelPredictions.csv"))
    mean_pred_df.to_csv(join(out_path,
                             "ensembleClfMeanProba.csv"))
    binary_pred_df.to_csv(join(out_path,
                               "urgentProba.csv"))


def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i", dest="x_path", type=str,
                               help="input image directory")
    module_parser.add_argument("-o", dest="out_path", type=str,
                               help="output dir path")
    module_parser.add_argument("-m", dest="ensemble_path",
                               type=str,
                               help="path to model directory")
    return module_parser


def main(x_path, out_path, ensemble_path):
    """
    takes a directory of new images and runs inference with the classifier
    :param x_path: path to directory of new images (str)
    :param out_path: path to directory for predictions (str)
    :param ensemble_path: path to metaClf directory (str)
    :return: None
    """

    """############################################################################
                        0. Preprocess Data
    ############################################################################"""
    set_image_data_format('channels_last')
    img_type_dict = get_class_labels(intKey=True)

    model_pred_dict = {}
    # generate predictions for each individual model
    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    for model_name in models:
        # chose which preprocessing method to use by detecting the model
        preprocess_input = get_preprocess(model_name)
        if model_name == "VGG16":
            res = 224
        else:
            res = 299
        new_size = (res, res, 3)
        img_data, img_names = preprocess_imgs(x_path, new_size)
        img_data = preprocess_input(img_data)

        """############################################################################
                            0. load model & predict
        ############################################################################"""
        # load model
        model, model_dir_path = load_model(model_name, ensemble_path)
        # run inference
        y_pred = model.predict(img_data,
                               batch_size=1,
                               verbose=1)
        # save intermediate predictions to csv
        y_pred_df = save_model_results(model_name, y_pred,
                                       img_names, model_dir_path,
                                       img_type_dict)
        model_pred_dict[model_name] = y_pred_df
    save_predictions(model_pred_dict, models, img_names, out_path)


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main(args.x_path,
             args.out_path,
             args.ensemble_path)
        print('predict.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
