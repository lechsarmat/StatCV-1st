import os
import io

from glob import glob
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

def load_standards(dirpath):
    """Loads png images of standards.
       >>> load_standards(0)
       Traceback (most recent call last):
        ...
       TypeError: dirpath must be string
    """
    if type(dirpath) != str:
        raise TypeError("dirpath must be string")
    filelist = glob(f'{dirpath}/*.png')
    if len(filelist) == 0:
        raise ValueError("png files not found")
    standard_dict = {}

    for i in range(len(filelist)):
        image = (np.array(Image.open(filelist[i])) / 255)
        if len(image.shape) != 3:
            raise ValueError("wrong dimensionality")
        else:
            image = (image[:,:,0] > 0.5).astype(int)
        if i == 0:
            h, w = image.shape
        elif image.shape != (h,w):
            raise ValueError("images must have the same shape")
        tempname = os.path.basename(filelist[i])[:-4]
        standard_dict[tempname] = image

    return standard_dict

if __name__ == "__main__":
    import doctest
    doctest.testmod()