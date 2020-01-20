import image_slicer

for i in range(10):
    image_slicer.slice('2D_plane_' + str(i) + ".png", 4)
