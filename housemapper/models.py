__author__ = 'lluiscanet'

import utils
import pickle
import numpy as np
import cv2
import logging

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import cross_val_score

type_dict = {'thatched': 0, 'iron': 1}


class ImageAnalysisModel:
    """
    General satellite image processing model
    """

    def __init__(self, name, train_colect, image_dir, model):
        self.name = str(name)
        self.data = train_colect
        self.image_dir = image_dir
        self.labels, self.features, self.clf = self.train(model)

    def train(self, model):
        logging.info('Training model ' + self.name)
        labels, features = self.get_labels_features()
        if features:
            return labels, features, model.fit(features, labels)
        else:
            raise Exception('No features available')

    def get_labels_features(self):
        tot_labels = list()
        tot_features = list()
        for row in self.data.iterrows():
            logging.info('Training image ' + str(row[0]))
            img = cv2.imread(self.image_dir + row[1].image)
            if img is not None:
                labels, features = self.row_to_features(img, row[1])
                tot_labels.extend(labels)
                tot_features.extend(features)
            else:
                print 'Image not found ' + row[1].image
        return tot_labels, tot_features

    def bulk_apply(self, df, image_folder, col_name):
        results = list()
        for row in df.iterrows():
            file_name = image_folder + row[1].image
            logging.info('%d - Processing image %s' % (row[0], file_name))
            result = self.apply(file_name)
            results.append(result)
        df[col_name] = results
        return df

    def apply(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        feature = self.image_to_features(img)
        return self.clf.predict([feature])[0]

    def row_to_features(self, img, row):
        """ Implement """
        return None, None

    def image_to_features(self, img):
        """ Implement """
        return None

    def save_model(self, file_name):
        file_save = open(file_name, "w")
        pickle.dump(self, file_save)
        file_save.close()


class RoofClassifierModel(ImageAnalysisModel):
    """
    Contains a model to predict roof types
    """

    def __init__(self, name, train_collect, image_dir, model=None):
        logging.info('Initialize Roof Classification Model')
        if model is None:
            model = RandomForestClassifier(n_estimators=50)
        ImageAnalysisModel.__init__(self, name, train_collect, image_dir, model)

    def row_to_features(self, img, row):
        labels = list()
        features = list()
        for roof in row.roofs:
            labels.append(type_dict[roof['type']])
            roof_patch = utils.crop_from_center(img, roof['x'], roof['y'])
            features.append(utils.color_feat(roof_patch))
        return labels, features

    def image_to_features(self, img):
        features = utils.color_feat(img)
        return self.clf.predict_proba(features)

    def get_clf_performance(self, cv=10):
        return cross_val_score(self.clf, self.features, self.labels, cv=cv)


class RegressionModel(ImageAnalysisModel):
    """
    General regression model
    """
    def __init__(self, name, train_collect, image_dir, roof_model, iron_template, grass_template, model):
        self.roof_model = roof_model
        self.iron_temp = iron_template
        self.grass_temp = grass_template
        ImageAnalysisModel.__init__(self, name, train_collect, image_dir, model)

    def get_mean_absolute_error(self):
        oob_pred = self.clf.oob_prediction_
        differ = oob_pred - self.labels
        return np.abs(differ).mean()

    def predic(self, image):
        features = self.image_to_features(image)
        self.clf.predict(features)

    def detect_objects(self, img, template, thres):
        #Conver to gray scale
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Match Template
        temp_match = cv2.matchTemplate(img_grey, template, 1)
        #Normalize scores
        temp_match = temp_match-temp_match.min()
        temp_match = 1 - temp_match/temp_match.max()
        #Apply Threshold
        retval, dst = cv2.threshold(temp_match, thres, 1, 0)
        dst = dst.astype(np.uint8)
        #Extract centroid of connected components
        contours, hierarchy = cv2.findContours(dst, 1, 2)
        if len(contours) > 0:
            m_df = DataFrame([cv2.moments(cont) for cont in contours])
            m_df['x'] = (m_df['m10']/m_df['m00']) + int(template.shape[0]/2)
            m_df['y'] = (m_df['m01']/m_df['m00']) + int(template.shape[1]/2)
            m_df.dropna(subset=['x', 'y'], inplace=True)
            return m_df
        else:
            return DataFrame()


class RoofCountRegressionModel(RegressionModel):
    """
    Regression Model for Roof Counts
    """

    def __init__(self, name, train_collect, image_dir, iron_template, grass_template, row_count, model=None):
        logging.info('Initialize Roof Count Regression Model')
        if model is None:
            model = RandomForestRegressor(n_estimators=50, oob_score=True)
        self.row_count = row_count
        RegressionModel.__init__(self, name, train_collect, image_dir, None, iron_template, grass_template, model)

    def row_to_features(self, img, row):
        label = 0
        if row[self.row_count] > 0:
            label = row[self.row_count]
        features = self.image_to_features(img)
        return [label], [features]

    def image_to_features(self, img):
        c_feat = utils.color_feat(img)

        #Iron
        thres_iron = np.linspace(0.3, 1.0, 14, endpoint=False)
        iron_feat = np.array([self.get_threshold_features(img, self.iron_temp, thres)
                              for thres in thres_iron])

        #Grass
        thres_grass = np.linspace(0.8, 1.0, 14, endpoint=False)
        grass_feat = np.array([self.get_threshold_features(img, self.grass_temp, thres)
                              for thres in thres_grass])

        #Total features
        features = np.concatenate([iron_feat, grass_feat, c_feat]).tolist()
        return features

    def get_threshold_features(self, img, template, thres):
        moments = self.detect_objects(img, template, thres)
        num_obj = moments.shape[0]
        return num_obj


class RoofRatioRegressionModel(RegressionModel):
    """
    Regression Model for Roof Ratio
    """

    def __init__(self, name, train_collect, image_dir, roof_model, iron_template, grass_template, model=None):
        logging.info('Initialize Roof Ratio Regression Model')
        if model is None:
            model = RandomForestRegressor(n_estimators=50, oob_score=True)
        RegressionModel.__init__(self, name, train_collect, image_dir, roof_model, iron_template, grass_template, model)

    def row_to_features(self, img, row):
        label = 0
        if row.total > 0:
            label = row.number_iron*1.0/row.total
        features = self.image_to_features(img)
        return [label], [features]

    def image_to_features(self, img):
        if self.roof_model is None:
            return np.zeros((80,))
        c_feat = utils.color_feat(img)

        #Iron
        thres_iron = np.linspace(0.3, 1.0, 14, endpoint=False)
        iron_feat = np.array([self.get_threshold_features(img, self.iron_temp, thres, self.roof_model)
                              for thres in thres_iron])
        iron_feat = np.concatenate(np.transpose(iron_feat))

        #Grass
        thres_grass = np.linspace(0.8, 1.0, 14, endpoint=False)
        grass_feat = np.array([self.get_threshold_features(img, self.grass_temp, thres, self.roof_model)
                              for thres in thres_grass])
        grass_feat = np.concatenate(np.transpose(grass_feat))

        #Total features
        features = np.concatenate([iron_feat, grass_feat, c_feat]).tolist()
        return features

    def get_threshold_features(self, img, template, thres, roof_classifier):
        moments = self.detect_objects(img, template, thres)
        num_obj = moments.shape[0]
        if num_obj == 0:
            return num_obj, 0
        else:
            #Get classfication scores
            patches = [utils.crop_from_center(img, x, y) for x, y in zip(moments['x'], moments['y'])]
            #Probabilities of being iron
            scores = [roof_classifier.image_to_features(patch)[0][1] for patch in patches]
            mean_sc = np.mean(scores)
            return num_obj, mean_sc
