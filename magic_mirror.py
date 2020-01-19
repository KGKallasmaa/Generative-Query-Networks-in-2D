from PIL import Image

from view_pointer import generate_3_elem, get_view, get_cameras_batch
import numpy as np
import matplotlib.pyplot as plt


def mirror_mirror_on_the_wall(views):
    # We're assuming that the canvas ins black
    results = np.zeros((300, 300, 3), dtype=np.uint8)

    backcround = np.array([0, 0, 0])

    # I = currect view
    for i in range(len(views)):
        # J = row
        j = 0
        while j <len(views[i][0]):
            if np.array_equal(views[i][0][j], backcround):
                views[i][0][j] = backcround
                # todo: fix this
            else:
                print("result", results[i][j])
                results[i][j] = views[i][0][j]
                #  print(views[i][j])
                if i == 0:
                    b = 4
                elif i == 1:
                    a = 3
                elif i == 2:
                    a = 3
                else:
                    a = 3
            j += 3


    """
            view 1
      view 4      view 2
            view 3
    """

    # for view in views:
    #    print(row)
    #   print("hi")
    return results


def save_as_img(array, i):
    # for testing
    im = Image.fromarray(array)
    file_name = "./delete/plane" + str(i) + ".png"
    im.save(file_name)


if __name__ == '__main__':
    pic = generate_3_elem(300, 300)
    save_as_img(pic, 0)
    view3 = get_view(pic, spread=300)
    view2 = get_view(pic, spread=200)
    view1 = get_view(pic, spread=100)
    view0 = get_view(pic, spread=0)
    views = [view0, view1, view2, view3]
    result = mirror_mirror_on_the_wall(get_cameras_batch())
    save_as_img(result, 1)
