# -*- coding: utf-8 -*-
import cv2
import sys
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def to_degrees(mat):
    return mat * (180/np.pi)

def to_radians(mat):
    '''Use this to convert array from degrees to radians'''
    return mat * (np.pi/180)

def to_coordinates(radius, degrees):
    return [ [ radius * np.cos(degree), radius * np.sin(degree) ] for degree in degrees ]

def color_dispenser(n_colors = 3):
    '''Generate a grid of random RBG colors, default number of random n_colors = 3.
    Also adds some threshold value so that the shapes are not too close to black'''
    return ( np.random.rand(n_colors, 3) * 220 + 30 ).astype(np.uint8)

def matter(x, y):
    '''Create a matrix with dimensions.'''
    return np.zeros((x, y, 3), dtype=np.uint8)

def get_rect(mat, x, y, x_end, y_end, color=(0, 0, 0)):
    '''Place a rectangle on matrix defined by x to x_end, y to y_end with color'''
    mat[y:y_end, x:x_end] = color
    return mat

# NOTE: This thing must be replaced with cv2.ellipsis if we want mutatable dimension shapes
def get_circled(mat, x, y, x_end, y_end, color=(0, 0, 0)):
    '''Place a circle on matrix defined by bounding rectangle x to x_end, y to y_end with color'''
    deltas = (x + np.int( ( x_end - x )/2 ), y + np.int( ( y_end - y )/2 ))
    # crazy cv2 data formats....
    return cv2.circle(mat, deltas, np.int( ( x_end - x  ) / 2 ),
                 color = (int( color[0] ), int( color[1] ), int( color[2] )), thickness = -1)

def get_triangled(mat, x, y, x_end, y_end, color=(0, 0, 0)):
    '''Place a triangle on matrix defined by bounding rectangle x to x_end, y to y_end with color'''
    points = np.array( [(x + np.int( ( x_end - x )/2 ), y), (x, y_end), (x_end, y_end)] )
    return cv2.drawContours(mat, [ points ], 0,
                       color = (int( color[0] ), int( color[1] ), int( color[2] )), thickness = -1)

def generate_3_elem(x_dim, y_dim):
    '''Generate a sample picture with three random shapes and colors. Size as parameter'''
    pic = matter(x_dim, y_dim)
    colors = color_dispenser()
    locations = np.asarray([[110, 50, 190, 130], # upper
                            [170, 170, 250, 250], # right down
                            [50, 170, 130, 250]]) # left down
    np.random.shuffle(locations)

    ## white background if you need it
    # pic = get_rect(pic, 0, 0, x_dim, y_dim, color = [255, 255, 255])
    pic = get_rect(pic, *locations[0], color = colors[0])
    pic = get_circled(pic, *locations[1], color = colors[1])
    return get_triangled(pic, *locations[2], color = colors[2]), colors

def get_view(mat, colors, tolerance = 2, spread = 0):
    '''Get the 1D array of the first non-background pixel with certain tolerance.
    Also you can spread the indice x times to create a pseudo-2D view.'''
    last = []
    report = np.zeros((max(mat.shape), 3), dtype = np.uint8)
    for y in range(len(mat)):
        for x in range(len(mat[y])):
            for col in colors:
                if np.allclose(mat[x, y], col, atol = tolerance):
                    report[y] = mat[x, y]
                    break
    if spread == 0:
        return [ report ]
    else:
        return [ report ] * spread

def view_pointer(mat, colors, yaws = [], tolerance = 2, spread = 0):
    '''Generate the views from a list of angles and return views in list.
    See further at get_view()'''
    views = []
    for yaw in yaws:
        # calculate the yaw from degree
        to_degree = yaw
        # rotates the picture and reshape = F holds it in shape.
        views.append(get_view(ndimage.rotate(mat, to_degree, reshape =False),
                              colors = colors,
                              tolerance = tolerance,
                              spread = spread))
    return views

# pic, colors = generate_3_elem(300, 300)
# view = get_view(pic, colors, spread = 0)
# plt.imshow(pic)
# plt.imshow(ndimage.rotate(pic, 60, reshape =False))
# view_trans = get_view(ndimage.rotate(pic, 60, reshape =False), colors)
# degrees = np.random.randint(360, size = 1) - 180
# cameras = view_pointer(pic, colors, yaws = degrees)
# plt.imshow(cameras[0])


# then look how it behaves in the gqn.
if __name__ == '__main__':
    # run the code and generate the pt's
    cameras_batch = []
    print('started processing')
    for batch in range(16):
        print("Batch", batch, "out of 64 started.", end = '\r')
        pic, colors = generate_3_elem(300, 300)
        # plt.imshow(pic)
        # generate degrees
        degrees = np.random.randint(360, size = 15) - 180
        # use "spread" to generate pseudo2D pictures
        cameras = view_pointer(pic, colors = colors, yaws = degrees, spread = 0)
        # convert together some viewpoint coordinates for the 15 views
        viewpoints = np.zeros((15, 5))
        viewpoints[:, 3] = to_radians( degrees )
        viewpoints[:, 0:2] = to_coordinates(150, degrees)
        # mesh them together into context
        context = [cameras, viewpoints]
        cameras_batch.append(context)
    torch.save(cameras_batch, "train_1d_16block_01.pt")
