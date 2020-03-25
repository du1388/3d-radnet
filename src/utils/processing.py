import numpy as np

def FindCenterBounds(size, target):

    difference = target/2 - size/2
    lower_bound = difference
    upper_bound = difference + size

    return int(lower_bound), int(upper_bound)

def CenterImage(img_array, target_shape):
    img_dtype = img_array.dtype

    lb1, ub1 = FindCenterBounds(img_array.shape[0],target_shape[0])
    lb2, ub2 = FindCenterBounds(img_array.shape[1],target_shape[1])
    lb3, ub3 = FindCenterBounds(img_array.shape[2],target_shape[2])

    target_array = np.zeros(target_shape, dtype=img_dtype)
    target_array[lb1:ub1,lb2:ub2,lb3:ub3] = img_array

    return target_array

def CenterImage2D(img_array, target_shape):
    img_dtype = img_array.dtype

    lb1, ub1 = FindCenterBounds(img_array.shape[0],target_shape[0])
    lb2, ub2 = FindCenterBounds(img_array.shape[1],target_shape[1])

    target_array = np.zeros(target_shape, dtype=img_dtype)
    target_array[lb1:ub1,lb2:ub2] = img_array

    return target_array


def FitCenterBox(img_array, target_shape):

    # Find maximum size
    coord = list(img_array.shape) + list(target_shape)
    max_shape = [np.max(coord)]*3
    max_array = CenterImage(img_array, max_shape)

    # find coordinate
    lb1, ub1 = FindCenterBounds(target_shape[0],max_array.shape[0])
    lb2, ub2 = FindCenterBounds(target_shape[1],max_array.shape[1])
    lb3, ub3 = FindCenterBounds(target_shape[2],max_array.shape[2])

    return max_array[lb1:ub1,lb2:ub2,lb3:ub3]



