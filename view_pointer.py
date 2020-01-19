# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def to_degrees(mat):
    return mat * (180/np.pi)

def to_radians(mat):
    '''Use this to convert array from degrees to radians'''
    return mat * (np.pi/180)

def color_dispenser(n_colors = 3):
    '''Generate a grid of random RBG colors, default number of random n_colors = 3'''
    return ( np.random.rand(n_colors, 3) * 255 ).astype(np.uint8)

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
    return get_triangled(pic, *locations[2], color = colors[2])

def get_view(mat, background = [0, 0 ,0], tolerance = 5):
    '''Get the 1D array of the first non-background pixel with certain tolerance.'''
    report = np.zeros((max(mat.shape), 3))
    for y in range(len(mat)):
        for x in range(len(mat[y])):
           if not np.allclose(mat[x, y], background, rtol = tolerance):
               report[y] = mat[x, y]
               break
    return report

def view_pointer(mat, yaws = [], background = [0, 0, 0], tolerance = 5):
    '''Generate the views from a list of angles or yaws and return views in list'''
    views = []
    for yaw in yaws:
        # calculate the yaw from degree
        to_degree = yaw
        # rotates the picture and reshape = F holds it in shape.
        views.append(get_view(ndimage.rotate(mat, 60, reshape =False),
                              background = background,
                              tolerance = tolerance))
    return views

# one idea to solve the dimensionality problem is to project the 1D to square out.
# then look how it behaves in the gqn.
if __name__ == '__main__':
    # run the code and generate the pt's
    cameras_batch = []
    for batch in range(64):
        print("Batch", batch, "out of 64 started.", end = '\r')
        pic = generate_3_elem(300, 300)
        # plt.imshow(pic)
        # generate degrees
        degrees = np.random.randint(360, size = 15) - 180
        cameras = view_pointer(pic, yaws = degrees)
        # convert together some viewpoint coordinates
        viewpoints = [0, 0, 0, to_radians( degrees ), 0]
        context = [cameras, viewpoints]
        cameras_batch.append(context)
