__author__ = 'lluiscanet'

import housemapper as hm

# df = hm.download_division_images('BORO')
# df.to_csv('../data/boro_images.csv')
# print(len(df))

df = hm.download_division_images('KAREMO')
df.to_csv('../data/karemo_images.csv')
print(len(df))

# df = hm.download_division_images('URANGA')
# df.to_csv('../data/uranga_images.csv')
# print(len(df))
