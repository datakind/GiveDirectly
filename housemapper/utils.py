__author__ = 'lluiscanet'

import cv2
import logging
import shapefile
import numpy as np
import math
import pyproj

from urllib import urlretrieve
from shapely.geometry import Polygon, Point
from shapely.ops import transform
from functools import partial


def crop_from_center(img, x, y, size=14):
    """
    Crop image above specified roof position
    :param img: image to crop
    :param x: crop center position x
    :param y: crop center position y
    :return: cropped image
    """
    trans = int(size / 2)
    size = img.shape
    roof_patch = img[max(y - trans, 0):min(y + trans, size[0]), max(x - trans, 0):min(x + trans, size[1]), :]
    return roof_patch


def color_feat(patch, feat_num=8):
    """
    Extract color features from image patch
    :param patch: small patch of roof image
    :param feat_num: number of histograms bins per color
    :return: feature vector with normalized color histograms
    """
    color_f = list()
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([patch], [i], None, [feat_num], [0, 256])
        norm = histr / histr.sum()
        color_f.extend(norm.ravel().tolist())
    return color_f


def save_google_image(lat, lon, api_key, folder, name, height=400, width=400):
    """
    Get satellite image from google API
    :param lat: latitude
    :param lon: longitude
    :param api_key: Google API key
    :param folder: Image folder to store image
    :param name: Name of the file
    """
    url_pattern = 'http://maps.googleapis.com/maps/api/staticmap?center=%s,%s&zoom=19&size=%sx%s&' \
                  'maptype=satellite&sensor=true&key=%s'

    url = url_pattern % (lat, lon, height, width, api_key)
    image_name = "%s-%s-%s.png" % (name, lat, lon)
    fp = "%s/%s" % (folder, image_name)
    logging.info("Scraping %s,%s to %s" % (lat, lon, fp))
    urlretrieve(url, fp)
    return image_name


class ShapeHelper:
    def __init__(self, shape_file, lat_offset=0.0, lon_offset=0.0):
        self.sf = shapefile.Reader(shape_file)
        self.lat_offset = lat_offset
        self.lon_offset = lon_offset
        self.field_names = {self.sf.fields[i][0]: i-1 for i in range(1, len(self.sf.fields))}
        #Mercator projector
        self.merc_project = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),
            pyproj.Proj('+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs'))

    def filter_by_field(self, field_name, filt):
        """
        Given the name of the field return shapeRecords that satified the filter requirement
        :param field_name: Name of the field to match filter
        :param filt: filter requirement
        :return: list of shapeRecords
        """
        field_index = self.field_names[field_name]
        shape_records = [shape_record for shape_record in self.sf.shapeRecords()
                         if shape_record.record[field_index].lower() == filt.lower()]
        return shape_records

    def get_point_record_field(self, point, field_name):
        """
        Given a geographic point with 'lat' 'lon' attributes, find the shape that contains the point and return
        the content of the specified record field
        :param point: dict or dataframe row that contains 'lat', 'lon' attributes
        :param field_name: name of the field record attribute to extract
        :return: record attribute
        """
        coord_tuple = (point['lon'], point['lat'])
        for shape_record in self.sf.shapeRecords():
            if in_polygon_project(self.merc_project, [coord_tuple], self.offset_points(shape_record.shape.points))[0]:
                field_attr = shape_record.record[self.field_names[field_name]]
                logging.info(field_attr + ' ' + str(coord_tuple))
                return field_attr
        logging.info('No Attribute ' + str(coord_tuple))

    def get_shape_areas(self, record_key):
        """
        Get a dict of all the shape areas keyed by the record attribute specified in record_key
        :param record_key:
        :return:
        """
        areas = dict()
        for shape_record in self.sf.shapeRecords():
            key = shape_record.record[self.field_names[record_key]]
            area = polygon_project_area(self.merc_project, self.offset_points(shape_record.shape.points), key)
            areas[key] = area
        return areas

    def coord_in_field(self, field_name, filt, box_pix, zoom):
        """
        Obtain coordinates in a grid with box_pix resolution inside the shapes contained in the list specified by filt
        :param field_name: name of the field in the shape file that defines the filter
        :param filt: filter for the field in the shape file
        :param box_pix: width and height
        :param zoom: mercator zoom level to define the pixel resolution
        :return coordenate dictionary
        """
        shape_records = self.filter_by_field(field_name, filt)
        coords = list()
        for shape_record in shape_records:
            lat_lon_grid = get_shape_grid(shape_record, box_pix, zoom)
            coord_tuples = zip(lat_lon_grid[1].flatten(), lat_lon_grid[0].flatten())
            booleans = np.array(in_polygon_project(self.merc_project, coord_tuples, self.offset_points(shape_record.shape.points)))
            long_lat = np.array(coord_tuples)
            long_lat = long_lat[booleans]
            lat_lon_tuples = zip(long_lat[:, 1], long_lat[:, 0])
            patch_dicts =[ {'provname': shape_record.record[self.field_names['PROVNAME']],
                          'divname': shape_record.record[self.field_names['DIVNAME']],
                          'locname': shape_record.record[self.field_names['LOCNAME']],
                          'slname': shape_record.record[self.field_names['SLNAME']],
                          'lat': lat_lon_tuple[0],
                          'lon': lat_lon_tuple[1]} for lat_lon_tuple in lat_lon_tuples]
            coords.extend(patch_dicts)

        return coords

    def offset_points(self, points):
        """
        Offset points in shape file
        :param lat_offset: latitude offset
        :param lon_offset: longitude offset
        """
        if self.lat_offset > 0 or self.lon_offset > 0:
            arr = np.array(points)
            arr[:, 0] = np.array(points)[:, 0]+self.lon_offset
            arr[:, 1] = np.array(points)[:, 1]+self.lat_offset
            return arr.tolist()
        return points



def get_shape_grid(shape_record, box_pix, zoom):
    """
    Obtain a grid of latitude and longitude values
    :param shape_record: shape records that defines the boundaries of the grid
    :param box_pix: width and height of image patches
    :param zoom: mercator zoom level
    :return: two numpy arrays representing latitudes and longitudes in the grid
    """
    range1 = lambda start, end, width: range(start, end+width, width)
    bbox = shape_record.shape.bbox
    x_min, y_max = lat_lon_to_pixel(bbox[1], bbox[0], zoom)
    x_max, y_min = lat_lon_to_pixel(bbox[3], bbox[2], zoom)
    pixel_grid = np.meshgrid(range1(0, y_max-y_min, box_pix), range1(0, x_max-x_min, box_pix))
    lat_lon_grid = get_new_lat_lon_position(bbox[3], bbox[0], pixel_grid[0], pixel_grid[1], zoom)
    return lat_lon_grid


def get_outer_bbox(shape_records):
    """
    Given a list of shape records provide outer bounding box
    :param shape_records:
    :return: list[x_min, y_min, x_max, y_max]
    """
    bboxes = np.array([shape_record.shape.bbox for shape_record in shape_records])
    map(bboxes)
    return [bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max()]


def lat_lon_to_pixel(lat, lon, zoom):
    """
    Given latitude and longitude values, return the pixel location in mercator projection given zoom value
    :param lat: latitude
    :param lon: longitude
    :param zoom: zoom value
    :return: x and y pixel location
    """
    pixels_width = math.sqrt(4**zoom)*256

    x = (lon+180)*(pixels_width/360)

    # convert from degrees to radians
    latRad = lat*math.pi/180

    # get y value
    mercN = math.log(math.tan((math.pi/4)+(latRad/2)))
    y = (pixels_width/2)-(pixels_width*mercN/(2*math.pi))

    return int(x), int(y)


@np.vectorize
def get_new_lat_lon_position(lat_ini, lon_ini, lat_pix_offset, lon_pix_offset, zoom):
    #Mercator projection
    pixels_width = math.sqrt(4**zoom)*256

    x = (lon_ini+180)*(pixels_width/360) + lon_pix_offset

    # convert from degrees to radians
    latRad = lat_ini*math.pi/180;

    # get y value
    mercN = math.log(math.tan((math.pi/4)+(latRad/2)))
    y = (pixels_width/2)-(pixels_width*mercN/(2*math.pi)) + lat_pix_offset

    #Revert to get latitue and longitude
    lon = (x/(pixels_width/360))-180

    mercN_new = ((pixels_width/2-y)*2*math.pi)/pixels_width
    latRad_new = (math.atan(math.exp(mercN_new)) - math.pi/4)*2

    lat = latRad_new*180/math.pi

    return lat, lon


def in_polygon_project(project, points, polygon):
    """
    Project point and polygon and return true if point inside projection
    :param project:
    :param point: tuple of longitude latitude
    :param polygon: list of tuples or arrays of longitude latitudes defining the polygon
    :return: True or False
    """
    if len(points) > 1:
        logging.info('Evaluating %s points in polygon' % len(points))
    poly = Polygon(polygon)
    res = list()
    for point in points:
        p1 = Point(point)

        # Translate to spherical Mercator or Google projection
        poly_g = transform(project, poly)
        p1_g = transform(project, p1)
        res.append(poly_g.contains(p1_g))

    return res


def polygon_project_area(project, polygon, key=None):
    """
    Project point and polygon and return true if point inside projection
    :param project: Coordinate projection
    :param polygon: list of tuples or arrays of longitude latitudes defining the polygon
    :return: Area of polygon determined by projection
    """
    logging.info('Evaluating area of polygon %s' % key)
    poly = Polygon(polygon)


    # Translate to spherical Mercator or Google projection
    poly_g = transform(project, poly)
    return poly_g.area


