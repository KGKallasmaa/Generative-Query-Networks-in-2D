#!/usr/bin/env ipython

# testing scripts to be run in ipython shell for some data understanding

import data_reader as dr

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import torch
import cv2
import sys

def lookadoodle(orig, mod):
    '''See modified picture and original side-by-side'''
    plt.clf()
    plt.subplot(122),plt.imshow(mod)
    plt.title('image one'), plt.xticks([]), plt.yticks([])

    plt.subplot(121),plt.imshow(orig)
    plt.title('image two'), plt.xticks([]), plt.yticks([])

def lookadoodles(data, tetris_block):
    '''See tetris_block at all views'''
    plt.clf()
    for tetris_num in range(15):
        plt.subplot(3, 5, tetris_num + 1), plt.imshow(data[tetris_block][0][tetris_num])
        plt.title('image ' + str( tetris_num )), plt.xticks([]), plt.yticks([])

def lookatetris(data, tetris_num):
    '''See tetris_num-th view of all blocks'''
    plt.clf()
    for tetris_block in range(len(data)): # should be 64 blocks
        plt.subplot(8, 8, tetris_block + 1), plt.imshow(data[tetris_block][0][tetris_num])
        plt.xticks([]), plt.yticks([])

def save_views(data, tetris_block):
    '''Save views of same tetris block as png files'''
    for tetris_num in range(15):
        f = './block' + tetris_block + '_view_' + str( tetris_num ) + '.png'
        cv2.imwrite(f, data[tetris_block][0][tetris_num])

def save_tetrises(data, tetris_num):
    '''Save views of all tetris blocks in data as png files'''
    for tetris_block in range(len(data)): # should be 64 blocks
        save_views(data, tetris_block)

# look at the pt's of metzler 5 parts (from Laura)
# tetris_block is the nth block looked at different angles
# tetris_num is the nth viewpoint looked from
tetris_block = 6
tetris_num = 6
ptest = torch.load("./random_testing/train/501-of-900-01.pt")
print( len(ptest), len(ptest[0]), np.shape( ptest[0][0] ) )
lookatetris(ptest, tetris_num)
#lookadoodles(ptest, tetris_block)
print("\nfirst image loc values:", ptest[0][1][0],
      "\nsecond image loc values:", ptest[0][1][1])

ptest = torch.load("./random_testing/train/502-of-900-01.pt")
print( np.shape( ptest[0][0] ) )
lookatetris(ptest, tetris_num)
#lookadoodles(ptest, tetris_block)
print("\nfirst image loc values:", ptest[0][1][0],
      "\nsecond image loc values:", ptest[0][1][1])

# from this i conclude, that there are 15 views on one tetris object
# ptest[0..64:tetris_block][0:64x64 image->, 1:locational data->][0..15:views or locdata][64x64xrgb data]

ptest = torch.load("./our_2d_data/train/Pilt1_not0.pt")
print( np.shape( ptest[0] ) )
lookadoodle(ptest[0][0], ptest[0][1])
print("\nfirst image loc values:", ptest[1][0],
      "\nsecond image loc values:", ptest[1][1])

print('\nmetzler 5 parts in DataReader:')
dat = dr.DataReader("shepard_metzler_5_parts",
                 3,
                "./shepard_metzler_5_parts",
                 mode='test')

taskdata = dat.print_read(16)
tf.print(taskdata.query.context.cameras, output_stream=sys.stdout)

print('our dataset in DataReader:')
dat = dr.DataReader("our_2d_data",
                 2,
                "./our_2d_data/",
                 mode='train')

taskdata = dat.print_read(1)
tf.print(taskdata.query.context.cameras, output_stream=sys.stdout)

print('some file info')
dat_info = dat._dataset_info
files = dr._get_dataset_files(dat_info,
                               "train",
                               "./our_2d_data/")
print(files)
