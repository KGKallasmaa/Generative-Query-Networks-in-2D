#!/usr/bin/env ipython

import data_reader as dr
#import tfrecord-converter

dat = dr.DataReader("rooms_ring_camera",
                 3,
                 "/Users/allu/git/gqn-datasets/",
                 mode='test')

dat.read(8)
dat_info = dat._dataset_info
print(dat_info)

files = dr._get_dataset_files(dat_info,
                               "test",
                               "/Users/allu/git/gqn-datasets/")
