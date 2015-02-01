# -*- coding: utf-8 -*-

import pandas as pd
import json
import os

from time import sleep
from models import *
from utils import *
from ConfigParser import SafeConfigParser

# Initialize configuration
config_file = os.getenv('HM_CONFIG', '../config/config.ini')
config = SafeConfigParser()
config.read(config_file)

#Configure logging
_log_path = config.get('GENERAL_CONFIG', 'log_file')
_log_level = config.get('GENERAL_CONFIG', 'log_level')

if _log_path:
    logging.basicConfig(filename=_log_path, level=_log_level, datefmt='%Y-%m-%d %H:%M')
else:
    logging.basicConfig(level=_log_level, datefmt='%Y-%m-%d %H:%M')

#Initialize global variables
_models_dir = config.get('GENERAL_CONFIG', 'models_dir')


def load_training_data():
    # Load training data
    json_data = open(config.get('GENERAL_CONFIG', 'roof_train_json'))
    roof_data = [json.loads(json_line) for json_line in json_data]
    image_meta = pd.read_csv(config.get('GENERAL_CONFIG', 'roof_train_data'))
    roof_train = DataFrame(roof_data)
    roof_train['image_tag'] = roof_train.image.map(lambda name: name.strip().split('-')[0])
    roof_train['image'] = roof_train.image.map(lambda name: name.strip())

    # Merge Training data
    all_train = pd.merge(image_meta, roof_train, left_on='GDID', right_on='image_tag')
    return all_train


def load_templates():
    #Create templates
    grass_tmp = cv2.imread(config.get('GENERAL_CONFIG', 'roof_grass_template'), 0)
    iron_tmp = (np.ones((12, 12)) * 255).astype(np.uint8)
    return grass_tmp, iron_tmp


def bin_data(image_data, image_dir=None):
    """
    Bin images based on type
    :param image_data: Data frame with image metadata
    :return: dict of bins
    """
    file_model = open(config.get('GENERAL_CONFIG', 'models_dir') + 'cluster_model.pkl', "r")
    if image_dir is None:
        image_dir = config.get('GENERAL_CONFIG', 'image_dir')
    cluster_model = pickle.load(file_model)
    file_model.close()
    image_data['features'] = image_data.image.map(lambda im: image_to_features(image_dir+im))
    feat_array = np.array(image_data['features'].tolist())
    image_data['cluster'] = cluster_model.predict(feat_array)
    image_data['cl_distance'] = cluster_model.transform(feat_array).min(axis=1)
    keys = image_data.cluster.unique()
    gb = image_data.groupby('cluster')
    data_bins = dict()
    for key in keys:
        data_bins[str(key)] = gb.get_group(key)
    return data_bins

def image_to_features(image_path):
    logging.info('Clustering: Extracting features ' + image_path)
    img = cv2.imread(image_path)
    features = utils.color_feat(img)
    return features

# #Deprecated
# def bin_data(image_data):
#     """
#     Bin images based on type
#     :param image_data: Data frame with image metadata
#     :return: dict of bins
#     """
#     long_name = 'long'
#     if 'long' not in image_data.columns:
#         long_name = 'lon'
#     data_bins = dict()
#     wet_cond = (image_data.lat > 0.1085) & (image_data[long_name] < 34.3135)
#     hazy_cond = image_data.lat < 0.0115
#     data_bins['wet'] = image_data[wet_cond]
#     data_bins['dry'] = image_data[(~wet_cond) & (~hazy_cond)]
#     data_bins['hazy'] = image_data[(~wet_cond) & hazy_cond]
#     return data_bins


class TrainModelFactory:
    """
    Factory to generate trained models
    """
    def __init__(self, file_prefix, bins=None):
        self.file_prefix = file_prefix
        self.bins = bins
        self.image_dir = config.get('GENERAL_CONFIG', 'image_dir')

    def get_models(self):
        training_data = load_training_data()
        data_bins = bin_data(training_data)
        models = dict()
        if self.bins is None:
            self.bins = data_bins.keys()
        for binn in self.bins:
            try:
                model = self.init_model(str(binn), data_bins)
                model.save_model(_models_dir + self.file_prefix + str(binn) + ".pkl")
                models[binn] = model
            except Exception:
                pass # Ignore exception and continue
        return models


class TrainRoofClassifierFactory(TrainModelFactory):
    """
    Factory to generate trained models to classify the roofs
    """
    def __init__(self, file_prefix, binn=None):
        TrainModelFactory.__init__(self, file_prefix, binn)

    def init_model(self, binn, data_bins):
        return RoofClassifierModel(binn, data_bins[binn], self.image_dir)


class TrainRoofRegressionFactory(TrainModelFactory):
    """
    Factory to generate trained regression models
    """
    def __init__(self,  file_prefix, class_model_prefix):
        TrainModelFactory.__init__(self, file_prefix)
        self.roof_model_prefix = class_model_prefix
        self.roof_class_model = None
        templates = load_templates()
        self.iron_template = templates[0]
        self.grass_template = templates[1]

    def init_model(self, binn, data_bins):
        try:
            file_model = open(_models_dir + self.roof_model_prefix + binn + '.pkl', "r")
            self.roof_class_model = pickle.load(file_model)
            file_model.close()
        except IOError:
            try:
                factory = TrainRoofClassifierFactory(self.roof_model_prefix, [binn])
                roof_class_model = factory.get_models()[0]
                self.roof_class_model = roof_class_model
            except Exception:
                logging.info('Roof classifier not initialized')
        return self.create_model(binn, data_bins)


class TrainRoofRatioRegressionFactory(TrainRoofRegressionFactory):
    """
    Factory to generate trained regression models to estimate ratio of iron to grass roofs
    """
    def __init__(self,  file_prefix, class_model_prefix):
        TrainRoofRegressionFactory.__init__(self, file_prefix, class_model_prefix)

    def create_model(self, binn, data_bins):
        return RoofRatioRegressionModel(binn, data_bins[binn], self.image_dir, self.roof_class_model,
                                        self.iron_template, self.grass_template)


class TrainRoofCountRegressionFactory(TrainRoofRegressionFactory):
    """
    Factory to generate trained regression models to estimate number of roofs
    """
    def __init__(self,  file_prefix, class_model_prefix, row_count):
        self.row_count = row_count
        TrainRoofRegressionFactory.__init__(self, file_prefix, class_model_prefix)

    def create_model(self, binn, data_bins):
        return RoofCountRegressionModel(binn, data_bins[binn], self.image_dir,
                                        self.iron_template, self.grass_template, self.row_count)


def download_division_images(division_name):
    shape_helper = ShapeHelper(config.get('GENERAL_CONFIG', 'admin_shapefile'))
    box_pix = config.getint('GENERAL_CONFIG', 'box_pix')
    zoom = config.getint('GENERAL_CONFIG', 'zoom')
    api_key = config.get('GENERAL_CONFIG', 'api_key')
    google_images = config.get('GENERAL_CONFIG', 'google_images')
    coords = shape_helper.coord_in_field('DIVNAME', division_name, box_pix, zoom)
    images = list()
    print(len(coords))
    for coord in coords:
        div_loc_subl = '%s-%s-%s-%s' % (coord['provname'], coord['divname'], coord['locname'], coord['slname'])
        image_name = save_google_image(coord['lat'], coord['lon'], api_key, google_images,
                          div_loc_subl.replace('/', '_'), box_pix, box_pix)
        coord['image'] = image_name
        images.append(coord)
    df = DataFrame(images)
    df.set_index('image', inplace=True)
    df = df[['provname', 'divname', 'locname', 'slname', 'lat', 'lon']]
    return df






