import os

mode ='train'
data_set = 'pascal' #'coco'
base_data_root ='D:\ly\data/base_annotation/pascal/'

fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, mode)
split =0
base_path = os.path.join(base_data_root, mode, str(split))
