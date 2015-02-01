__author__ = 'lluiscanet'

import housemapper as hm
import numpy as np
import pickle

#Roof classification models
train_factory = hm.TrainRoofClassifierFactory('class_model_')
roof_class_models = train_factory.get_models()
for name in roof_class_models:
    score = roof_class_models[name].get_clf_performance()
    mean_score = np.mean(score)
    size = len(roof_class_models[name].labels)
    print 'Roof classification score %s and size %s' % (str(mean_score), str(size))

#Roof ratio models
train_factory = hm.TrainRoofRatioRegressionFactory('ratio_model_', 'class_model_')
roof_ratio_models = train_factory.get_models()
for name in roof_ratio_models:
    score = roof_ratio_models[name].get_mean_absolute_error()
    mean_score = np.mean(score)
    size = len(roof_ratio_models[name].labels)
    print '%s : Roof Ratio MAE %s and size %s' % (name, str(mean_score), str(size))

#Roof count models
train_factory = hm.TrainRoofCountRegressionFactory('count_model_', 'class_model_', 'total')
roof_count_models = train_factory.get_models()
for name in roof_count_models:
    score = roof_count_models[name].get_mean_absolute_error()
    mean_score = np.mean(score)
    size = len(roof_count_models[name].labels)
    print '%s : Roof Count MAE %s and size %s' % (name, str(mean_score), str(size))

#Roof count models
train_factory = hm.TrainRoofCountRegressionFactory('iron_count_model_', 'class_model_', 'number_iron')
roof_iron_count_models = train_factory.get_models()
for name in roof_iron_count_models:
    score = roof_iron_count_models[name].get_mean_absolute_error()
    mean_score = np.mean(score)
    size = len(roof_iron_count_models[name].labels)
    print '%s : Iron Count MAE %s and size %s' % (name, str(mean_score), str(size))


# file_model = open('../models/ratio_model_hazy.pkl', "r")
# saved_model = pickle.load(file_model)
# file_model.close()
#
# error = saved_model.get_mean_absolute_error()
# print str(error)

