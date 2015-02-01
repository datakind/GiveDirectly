__author__ = 'lluiscanet'
import pickle
import os
import pandas as pd
import housemapper as hm


def load_models(model_dir, model_prefix):
    model_files = [mod for mod in os.listdir(model_dir) if '.pkl' in mod and model_prefix == mod[:len(model_prefix)]]
    models = dict()
    for model_file in model_files:
        key = model_file.replace(model_dir, '').replace(model_prefix, '').replace('/', '').replace('.pkl', '')
        file_model = open(model_dir + model_file, "r")
        saved_model = pickle.load(file_model)
        file_model.close()
        models[key] = saved_model
    return models


def apply_models(models, data_bins, image_folder, col_name):
    dfs = list()
    for binn in data_bins:
        model = models[binn]
        df = model.bulk_apply(data_bins[binn], image_folder, col_name)
        dfs.append(df)
    return pd.concat(dfs)


#Evaluate roof ratios
image_data = pd.read_csv('../data/subset_classify_village_area_images.csv')
data_bins = hm.bin_data(image_data, '../data/classify_images/')
ratio_models = load_models('../models/', 'ratio_model_')
ratio_df = apply_models(ratio_models, data_bins, '../data/classify_images/', 'roof_ratio')
ratio_df.set_index('image', inplace=True)
ratio_df.to_csv('../data/subset_classify_village_area_image_results.csv')


#Evaluate roof counts
image_data = pd.read_csv('../data/subset_classify_village_area_image_results.csv')
data_bins = hm.bin_data(image_data, '../data/classify_images/')
count_models = load_models('../models/', 'count_model_')
count_df = apply_models(count_models, data_bins, '../data/classify_images/', 'roof_counts')
count_df.set_index('image', inplace=True)
count_df.to_csv('../data/subset_classify_village_area_image_results.csv')


#Evaluate iron roof counts
image_data = pd.read_csv('../data/subset_classify_village_area_image_results.csv')
data_bins = hm.bin_data(image_data, '../data/classify_images/')
iron_count_models = load_models('../models/', 'iron_count_model_')
iron_df = apply_models(iron_count_models, data_bins, '../data/classify_images/', 'iron_counts')
iron_df.set_index('image', inplace=True)
iron_df.to_csv('../data/subset_classify_village_area_image_results.csv')
