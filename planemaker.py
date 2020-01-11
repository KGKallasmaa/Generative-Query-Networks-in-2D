import random
import numpy as np
from PIL import Image
from tornado import concurrent


def get_random_color():
    colors = {
        "red": np.array([244, 0, 0], dtype=np.uint8),
        "blue": np.array([50, 82, 123], dtype=np.uint8),
        "green": np.array([113, 238, 184], dtype=np.uint8),
    }
    color, code = random.choice(list(colors.items()))
    return code


def get_canvas(x, y, color=None):
    """
    :param x: width of the canvas
    :param y: height of the canvas
    :param color: the we want the background to be in
    """

    if color is None:
        color = np.array([0, 0, 0], dtype=np.uint8)
    canvas = []

    for i in range(x):
        row = []
        for j in range(y):
            for el in color:
                row.append(el)
        canvas.append(np.array(row, dtype=np.uint8))

    return np.array(canvas)


def make_triangle(x, y, color):
    canvas = get_canvas(x, y, color)

    for i in range(x):
        for j in range(y):
            if j > i:
                canvas[i][j] = np.array(0, dtype=np.uint8)
    return canvas


def make_circle(r, color):
    """
    :param r:radius of the circle
    :param color: color that is used
    :return: a 2D circle
    """
    background = np.array([0, 0, 0], dtype=np.uint8)

    A = np.arange(-r, r + 1) ** 2
    dists = np.sqrt(A[:, None] + A)
    circle = (np.abs(dists - r) < 0.5).astype(int)

    canvas = get_canvas(len(circle), len(circle))



    for i in range(len(circle)):
        for j in range(len(circle[0])):
            if circle[i][j] == 0:
                canvas[i][j:j + 3] = background
            else:
                canvas[i][j:j + 3] = color

    return canvas


def make_an_object(x, y):
    """
    :param x: width of the object
    :param y: height of the object
    :return: the object
    """
    #types = ["triangle", "circle", "rectangle"]
    types = ["circle", "rectangle"]

    random_type = random.choice(types)
    random_color = get_random_color()

    if random_type == "triangle":
        return make_triangle(x, y, random_color)
    elif random_type == "circle":
        return make_circle(x, random_color)
    return get_canvas(x, y, random_color)


def region_is_empty(stat_row, end_row, start_column, end_column, plane):
    for i in range(len(plane)):
        if stat_row <= i < end_row:
            for j in range(len(plane[0])):
                if start_column <= j < end_column:
                    if plane[i][j] != 0:
                        return False
    return True


def add_to_plane(random_object, plane):
    random_start_row = random.randint(0, len(plane) - len(random_object))
    random_start_column = random.randint(0, len(plane[0]) - len(random_object[0]))

    end_row = len(random_object) + random_start_row
    end_column = len(random_object[0]) + random_start_column

    nr_of_tries = 20
    region_is_available = False
    while nr_of_tries > 0:
        nr_of_tries -= 1
        region_is_available = region_is_empty(random_start_row, end_row, random_start_column, end_column, plane)
        if region_is_available:
            nr_of_tries = 0

    if region_is_available:
        l = 0
        for i in range(random_start_row, end_row):
            k = 0
            for j in range(random_start_column, end_column):
                plane[i][j] = random_object[l][k]
                k += 1
            l += 1

    return plane


def make_single_plane_as_array(plane_info):
    """
    :param plane_info:[x,y] = [x = plane_width , y = plane_height]
    :return: single plane
    """
    MAX_NR_OF_OBJECTS = 3
    MAX_OBJECT_SIZE = 50
    MIN_OBJECT_SIZE = 25

    print("Making plane nr", plane_info[2])

    single_plane = get_canvas(plane_info[0], plane_info[1])
    random_nr_of_objects = random.randint(3, MAX_NR_OF_OBJECTS)

    for i in range(random_nr_of_objects):
        random_width = random.randint(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        random_length = random.randint(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        random_object = make_an_object(random_width, random_length)

        single_plane = add_to_plane(random_object, single_plane)

    return single_plane


def make_planes_as_arrays(x, y, n):
    planes = []
    for i in range(n):
        planes.append([x, y, i])

    multi_thread_planes = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        for result in executor.map(make_single_plane_as_array, planes):
            multi_thread_planes.append(result)
    return multi_thread_planes


def make_planes(x=300, y=100, n=10):
    """
    :param x: the width of our plane field
    :param y: the height of our plane field
    :param n: number of planes that will be generated
    :return: 
    """
    planes = make_planes_as_arrays(x, y, n)

    for i in range(len(planes)):
        im = Image.fromarray(planes[i])
        file_name = "2D_plane_" + str(i) + ".png"
        im.save(file_name)


if __name__ == '__main__':
    make_planes()
