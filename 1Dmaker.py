import numpy as np
import matplotlib.image as mpimg

#Siia tuleb kirjutada pildifaili nimi
img=mpimg.imread('2D_plane_0.png')

#Need peavad olema võrdsed - st pilt peab olema ruut!
print("PILDI PIKKUS (Y-telg): " + str(len(img)))
print("PILDI LAIUS (X-telg): " + str(len(img[0])))

print("TOP TO BOTTOM:")
top_bottom = [0] * len(m)
for i in m:
    q = 0
    for element in np.nditer(i):
        if element != 0 and top_bottom[q] == 0:
            top_bottom[q] = element.tolist()
        q = q+1
print(top_bottom)

#left_to_right
print("LEFT TO RIGHT:")
left_right = [0] * len(m)
for i in np.transpose(m):
    q = 0
    for element in np.nditer(i):
        if element != 0 and left_right[q] == 0:
            left_right[q] = element.tolist()
        q = q+1
print(left_right)


print("BOTTOM TO TOP:")
bottom_top = [0] * len(m)
for i in np.flip(m):
    q = 0
    for element in np.nditer(i):
        if element != 0 and bottom_top[q] == 0:
            bottom_top[q] = element.tolist()
        q = q+1
print(bottom_top)

print("RIGHT TO LEFT:")
right_left = [0] * len(m)
for i in np.flip(np.transpose(m)):
    q = 0
    for element in np.nditer(i):
        if element != 0 and right_left[q] == 0:
            right_left[q] = element.tolist()
        q = q+1
print(right_left)
