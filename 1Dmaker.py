import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch

import image_slicer

#img=mpimg.imread('trainimg_1.png')
img= cv2.imread('trainimg_1.png')

#top_to_bottom = [0] * len(img)

print("PILDI PIKKUS (Y-telg): " + str(len(img)))
print("PILDI LAIUS (X-telg): " + str(len(img[0])))
print(img)
print(np.transpose(img, axes=(0, 1, 2)))
cv2.imshow("Sõna", img)


#ABILIST on väljund mis ei võta arvesse kujundi värvi vaid lihtsalt selle olemasolu.


#nüüd võtame arvesse ka värve !!!
#m = np.matrix([[0, 0, 3], [1, 2, 0], [0, 1, 0]])
m = img
#print(m)
#print(img)
#TOP TO BOTTOM
print("TOP TO BOTTOM")
top_bottom = [[0, 0, 0]] * len(m)
#print(top_bottom)
for i in m:
    #print(i)
    q = 0
    for element in i:#np.nditer(i):
        #print(element)
        #print(element.tolist())
        if np.array_equal(element, [0,0,0]) == False and top_bottom[q] == [0, 0, 0]:
        #if element != 0 and top_bottom[q] == 0:
            top_bottom[q] = element.tolist()
        q = q+1
#print(top_bottom)

#left_to_right
print("LEFT TO RIGHT ALGAB")
left_right = [[0, 0, 0]] * len(m)
#print(left_right)
for i in np.transpose(m, axes= (0, 1, 2)):#np.nditer(m, flags=['external_loop']):
    #print(i)
    q = 0
    for element in i:
        if np.array_equal(element, [0,0,0]) == False and left_right[q] == [0, 0, 0]:
        #if element != 0 and left_right[q] == 0:
            left_right[q] = element.tolist()
        q = q+1
#print(left_right)


print("BOTTOM TO TOP ALGAB")
bottom_top = [[0, 0, 0]] * len(m)
#print(bottom_top)
for i in np.flip(m):#np.nditer(m, flags=['external_loop']):
    #print(i)
    q = 0
    for element in i:
        if np.array_equal(element, [0,0,0]) == False and bottom_top[q] == [0, 0, 0]:
        #if element != 0 and bottom_top[q] == 0:
            bottom_top[q] = element.tolist()
        q = q+1
#print(bottom_top)

print("RIGHT TO LEFT ALGAB")
right_left = [[0, 0, 0]] * len(m)
#print(right_left)
for i in np.flip(np.transpose(m, axes=(0, 1, 2))):#np.nditer(m, flags=['external_loop']):
    #print(i)
    q = 0
    for element in i:
        if np.array_equal(element, [0,0,0]) == False and right_left[q] == [0, 0, 0]:
        #if element != 0 and right_left[q] == 0:
            right_left[q] = element.tolist()
        q = q+1
#print(right_left)

final_product_yaw0 = [np.array([top_bottom, left_right, bottom_top, right_left]),np.array([1,0,0,0,0],[-0.5,0,0,0,0],[0,0,0,0,0],[0.5,0,0,0,0])]
print("A")
torch.save(final_product_yaw0, "Pilt1.pt")                      
#final_product_yaw0 = [[top_bottom, left_right, bottom_top, right_left],[[1,0,0,0,0],[-0.5,0,0,0,0],[0,0,0,0,0],[0.5,0,0,0,0]]

asi1 = np.array([top_bottom, left_right, bottom_top, right_left])
asi2 = np.array([[1,0,0,0,0],[-0.5,0,0,0,0],[0,0,0,0,0],[0.5,0,0,0,0]])
final_product_yaw0 = [asi1, asi2]

asi3 = np.array([[1,0,0,3.14,0],[-0.5,0,0,1.57,0],[0,0,0,0,0],[0.5,0,0,-1.57,0]])
final_procut_yawnot0 = [asi1, asi3]

#print("A")
torch.save(final_product_yaw0, "Pilt1.pt")
torch.save(final_procut_yawnot0, "Pilt1_not0.pt")  
print("SAIN VALMIS!")
