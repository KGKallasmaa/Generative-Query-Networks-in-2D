#!/usr/bin/env ipython

# testing scripts to be run in ipython shell for some data understanding

import data_reader as dr

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import torch
import cv2
import sys
import os, gzip

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
    for tetris_block in range(len(data)): # should be 64 blocks ie 8x8 grid
        plt.subplot(8, 8, tetris_block + 1), plt.imshow(data[tetris_block][0][tetris_num])
        plt.xticks([]), plt.yticks([])

def save_views(data, tetris_block):
    '''Save views of same tetris block as png files'''
    for tetris_num in range(15):
        f = './block' + str( tetris_block ) + '_view_' + str( tetris_num ) + '.png'
        cv2.imwrite(f, data[tetris_block][0][tetris_num])

def save_tetrises(data):
    '''Save views of all tetris blocks in data as png files'''
    for tetris_block in range(len(data)): # should be 64 blocks
        save_views(data, tetris_block)

def to_degrees(mat):
    return mat * (180/np.pi)

def to_radians(mat):
    return mat * (np.pi/180)

def coords_to_coords(radius, coords):
    return np.array([np.cos(coords[:,0]) * radius,
                       np.sin(coords[:,1]) * radius])

# look at the pt's of metzler 5 parts (from Laura)
# tetris_block is the nth block looked at different angles
# tetris_num is the nth viewpoint looked from
tetris_block = 0
tetris_num = 6

ptest = torch.load("./random_testing/train/501-of-900-01.pt")
# ptest = torch.load("./trash_vol2/test/train_2d_64block_10.pt")
ptest = torch.load("./2d_64block128_01.pt")

print( len(ptest), len(ptest[0]), np.shape( ptest[0][0] ), len(ptest[0][1][0]) )
# lookatetris(ptest, tetris_num)
lookadoodles(ptest, tetris_block)
print("\nimage loc values:", ptest[tetris_block][1][0],
      "\nimage loc values:", ptest[tetris_block][1][1],
      "\nimage loc values:", ptest[tetris_block][1][11],
      "\nimage loc values:", ptest[tetris_block][1][13])
yaws = ptest[0][1][:, 3]
yaws_degrees = ptest[0][1][:, 3] * (180/np.pi)
pitch = ptest[0][1][:, 4]
pitch_degrees = ptest[0][1][:, 4] * (180/np.pi)

locs = ptest[0][1][:, 0:3]
print(coords_to_coords(32, locs))

ptest = torch.load("./random_testing/train/502-of-900-01.pt")
print( len(ptest), len(ptest[0]), np.shape( ptest[0][0] ) )
lookatetris(ptest, tetris_num)
#lookadoodles(ptest, tetris_block)
print("\nfirst image loc values:", ptest[0][1][0],
      "\nsecond image loc values:", ptest[0][1][1])

# from this i conclude, that there are 15 views on one tetris object
# ptest[0..64:tetris_block][0:64x64 image->, 1:locational data->][0..15:views or locdata][64x64xrgb data]

ptest = torch.load("./our_2d_data/train/Pilt1_not0.pt")
print( len(ptest), np.shape( ptest[0] ) )
lookadoodle(ptest[0][0], ptest[0][1])
print("\nfirst image loc values:", ptest[1][0],
      "\nsecond image loc values:", ptest[1][1])


# Lets look ad the data_reader.py here with the shepard metzler data
print('\nmetzler 5 parts in DataReader:')
_NUM_CHANNELS = 3
# this is the camera dimension parameter length modifier - let's keep it at 5 :)
_NUM_RAW_CAMERA_PARAMS = 5
dat = dr.DataReader("shepard_metzler_5_parts",
                 3,
                "./shepard_metzler_5_parts",
                 mode='train')

file_names = dr._get_dataset_files(dat._dataset_info, 'train', "./shepard_metzler_5_parts")
filename_queue = tf.train.string_input_producer(file_names, seed=99)
reader = tf.TFRecordReader()
_, raw_data = reader.read_up_to(filename_queue, num_records=16)
feature_map = {
    'frames': tf.FixedLenFeature(
        shape=dat._dataset_info.sequence_size, dtype=tf.string),
    'cameras': tf.FixedLenFeature(
        shape=[dat._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
        dtype=tf.float32)
}
example = tf.parse_example(raw_data, feature_map)
raw_pose_params = example['cameras']
raw_pose_params = tf.reshape(
    raw_pose_params,
    [-1, dat._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
pos = raw_pose_params[:, :, 0:3]
yaw = raw_pose_params[:, :, 3:4]
pitch = raw_pose_params[:, :, 4:5]
cameras = tf.concat(
    [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
print(pos, yaw, pitch)
print(cameras)

taskdata = dat.print_read(16)
tf.print(taskdata.query.context.cameras, output_stream=sys.stdout)


# Here we look at our own data...
print('our dataset in DataReader:')
dat = dr.DataReader("our_2d_data",
                 2,
                "./trash_vol2/",
                 mode='train')

taskdata = dat.print_read(16)
tf.print(taskdata.query.context.cameras, output_stream=sys.stdout)

cameras = dat._preprocess_cameras()

_NUM_CHANNELS = 3
# this is the camera dimension parameter length modifier - let's keep it at 5 :)
_NUM_RAW_CAMERA_PARAMS = 5
print('some file info')
dat_info = dat._dataset_info
files = dr._get_dataset_files(dat_info,
                               "train",
                               "./our_2d_data/")
print(files)

file_names = dr._get_dataset_files(dat._dataset_info, 'train', "./trash")
filename_queue = tf.train.string_input_producer(file_names, seed=99)
reader = tf.TFRecordReader()

_, raw_data = reader.read_up_to(filename_queue, num_records=16)
feature_map = {
    'frames': tf.FixedLenFeature(
        shape=dat._dataset_info.sequence_size, dtype=tf.string),
    'cameras': tf.FixedLenFeature(
        shape=[dat._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
        dtype=tf.float32)
}
example = tf.parse_example(raw_data, feature_map)
raw_pose_params = example['cameras']
raw_pose_params = tf.reshape(
    raw_pose_params,
    [-1, dat._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
pos = raw_pose_params[:, :, 0:3]
yaw = raw_pose_params[:, :, 3:4]
pitch = raw_pose_params[:, :, 4:5]
cameras = tf.concat(
    [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
print(pos, yaw, pitch)
print(cameras)
# save the tetrispics

scene_path = "./trash_vol2/test/train_2d_64block_10.pt.gz"

data = torch.load(gzip.open(scene_path, "r"))
images, viewpoints = list(zip(*data))

images = np.stack(images)
viewpoints = np.stack(viewpoints)

# uint8 -> float32
images = images.transpose(0, 1, 4, 2, 3)
images = torch.FloatTensor(images)/255
