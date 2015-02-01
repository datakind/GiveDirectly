__author__ = 'lluiscanet'

import pandas as pd

from housemapper import ShapeHelper
from pandas import DataFrame

lat_offset = 0.0021
lon_offset = 0.0019

def append_village_areas(divname):
    im_vil = pd.read_csv('../data/%s_village_images.csv' % divname.lower())
    shape_helper = ShapeHelper('../data/shapefiles/fixed_village_shapefiles/%s/%s.shp' % (divname.lower(), divname.lower()),
                               lat_offset, lon_offset)
    areas = shape_helper.get_shape_areas('village')
    areas_df = DataFrame(areas, index=['area'])
    areas_df = areas_df.transpose()
    areas_df.reset_index(inplace=True)
    areas_df.rename(columns={'index': 'village'}, inplace=True)
    im_vil_areas = pd.merge(im_vil, areas_df, how='left')
    im_vil_areas.set_index('image', inplace=True)
    im_vil_areas.to_csv('../data/%s_village_areas_images.csv' % divname.lower())

def append_village_name(divname):
    im_ref = pd.read_csv('../data/%s_images.csv' % divname.lower())
    shape_helper = ShapeHelper('../data/shapefiles/fixed_village_shapefiles/%s/%s.shp' % (divname.lower(), divname.lower()),
                               lat_offset, lon_offset)
    im_ref['village'] = im_ref.apply(lambda row: shape_helper.get_point_record_field(row, 'village'), 1)
    im_ref.set_index('image', inplace=True)
    im_ref.to_csv('../data/%s_village_images.csv' % divname.lower())

#Append village names to image record files
append_village_name('boro')
append_village_name('uranga')
append_village_name('karemo')

#Append village areas to shape image record files
append_village_areas('boro')
append_village_areas('uranga')
append_village_areas('karemo')
