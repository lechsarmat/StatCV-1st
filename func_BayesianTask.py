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

def noise_generator(X, p):
    """Generates Bernoulli noise.
       >>> noise_generator(np.array([[1]]), 0.)[0]
       array([[0]])
       >>> noise_generator(np.array([[1]]), '0')
       Traceback (most recent call last):
        ...
       TypeError: p must be float
       >>> noise_generator(np.array([[1]]), 1.1)
       Traceback (most recent call last):
        ...
       ValueError: p must be between 0 and 1
       >>> noise_generator('0', 0.1)
       Traceback (most recent call last):
        ...
       TypeError: X must be numpy array
       >>> noise_generator(np.array([1]), 0.1)
       Traceback (most recent call last):
        ...
       ValueError: wrong dimensionality of X
       >>> noise_generator(np.array([[1.]]), 0.1)
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements of X
       >>> noise_generator(np.array([[-1]]), 0.1)
       Traceback (most recent call last):
        ...
       ValueError: elements of X must be equal to 0 or 1
    """
    if type(p) != float:
        raise TypeError("p must be float")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if type(X) != np.ndarray:
        raise TypeError("X must be numpy array")
    if len(X.shape) != 2:
        raise ValueError("wrong dimensionality of X")
    if X.dtype != 'int':
        raise TypeError("wrong type of elements of X")
    if not np.all(np.logical_or(X == 0, X == 1)):
        raise ValueError("elements of X must be equal to 0 or 1")
    h, w = X.shape
    Y = np.random.binomial(1, p, (h, w))
    return Y, (X + Y) % 2

def bayes_classifier(noisy_image, standards, p):
    """Solves the Bayesian task.
       >>> bayes_classifier(np.array([[1]]), {'0':np.array([[0]]), '1':np.array([[1]])}, 0.1)
       '1'
       >>> bayes_classifier(np.array([[1]]), {'0':np.array([[0]]), '1':np.array([[1]])}, '0')
       Traceback (most recent call last):
        ...
       TypeError: p must be float
       >>> bayes_classifier(np.array([[1]]), {'0':np.array([[0]]), '1':np.array([[1]])}, 1.1)
       Traceback (most recent call last):
        ...
       ValueError: p must be between 0 and 1
       >>> bayes_classifier(np.array([[1]]), [np.array([[0]]), np.array([[1]])], 0.1)
       Traceback (most recent call last):
        ...
       TypeError: standards must be dictionary
       >>> bayes_classifier(np.array([[1]]), {}, 0.1)
       Traceback (most recent call last):
        ...
       ValueError: standards must contain images
       >>> bayes_classifier('0', {'0':np.array([[0]]), '1':np.array([[1]])}, 0.1)
       Traceback (most recent call last):
        ...
       TypeError: images must be numpy arrays
       >>> bayes_classifier(np.array([1]), {'0':np.array([[0]]), '1':np.array([[1]])}, 0.1)
       Traceback (most recent call last):
        ...
       ValueError: wrong dimensionality of images
       >>> bayes_classifier(np.array([[1.]]), {'0':np.array([[0]]), '1':np.array([[1]])}, 0.1)
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements of images
       >>> bayes_classifier(np.array([[1]]), {'0':np.array([[0, 1]]), '1':np.array([[1]])}, 0.1)
       Traceback (most recent call last):
        ...
       ValueError: images must have the same shape
       >>> bayes_classifier(np.array([[1]]), {'0':np.array([[0]]), '1':np.array([[-1]])}, 0.1)
       Traceback (most recent call last):
        ...
       ValueError: elements of images must be equal to 0 or 1
    """
    if type(p) != float:
        raise TypeError("p must be float")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if type(standards) != dict:
        raise TypeError("standards must be dictionary")
    if len(standards) == 0:
        raise ValueError("standards must contain images")
    if type(noisy_image) != np.ndarray:
        raise TypeError("images must be numpy arrays")
    if len(noisy_image.shape) != 2:
        raise ValueError("wrong dimensionality of images")
    if noisy_image.dtype != 'int':
        raise TypeError("wrong type of elements of images")
    if not np.all(np.logical_or(noisy_image == 0, noisy_image == 1)):
        raise ValueError("elements of images must be equal to 0 or 1")
    h, w = noisy_image.shape
    for standard in standards.values():
        if type(standard) != np.ndarray:
            raise TypeError("images must be numpy arrays")
        if len(standard.shape) != 2:
            raise ValueError("wrong dimensionality of images")
        if standard.dtype != 'int':
            raise TypeError("wrong type of elements of images")
        if standard.shape != (h,w):
            raise ValueError("images must have the same shape")
        if not np.all(np.logical_or(standard == 0, standard == 1)):
            raise ValueError("elements of images must be equal to 0 or 1")
    key_list = list(standards.keys())
    standards_num = len(key_list)
    res = np.zeros((standards_num,))

    for i in range(standards_num):
        xor_sum = noisy_image + standards[key_list[i]]
        temp_val = np.sum(np.log((1 - p) ** ((1 + xor_sum) % 2)))
        res[i] = np.sum(np.log(p ** (xor_sum % 2))) + temp_val

    return key_list[np.argmax(res)]

def plot_task(or_key, standards, p, savepath = None):
    """Shows the result of solving the task.
       >>> plot_task(0, {'0':np.array([[0]]), '1':np.array([[1]])}, 0.1)
       Traceback (most recent call last):
        ...
       TypeError: or_key must be string
       >>> plot_task('0', [np.array([[0]]), np.array([[1]])], 0.1)
       Traceback (most recent call last):
        ...
       TypeError: standards must be dictionary
       >>> plot_task('2', {'0':np.array([[0]]), '1':np.array([[1]])}, 0.1)
       Traceback (most recent call last):
        ...
       ValueError: or_key must be in standards
    """
    if type(or_key) != str:
        raise TypeError("or_key must be string")
    if type(standards) != dict:
        raise TypeError("standards must be dictionary")
    if or_key not in standards:
        raise ValueError("or_key must be in standards")
    X = standards[or_key]
    Y, Z = noise_generator(X, p)
    gs_key = bayes_classifier(Z, standards, p)

    im_list = [X, Y, Z, standards[gs_key]]
    lb_list = [f'Original: {or_key}', 'Noise',
                'Generated', f'Guess: {gs_key}']

    for i in range(4):
        plt.figure(figsize = (4, 4))
        plt.imshow(im_list[i], cmap = 'binary')
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(lb_list[i], fontsize = 24)

        buf = io.BytesIO()
        plt.savefig(buf, format = 'png')
        plt.close()
        im_cur = Image.open(buf)
        if i == 0:
            h, w = im_cur.height, im_cur.width
            im_res = Image.new('RGB', (4 * w, h))
        im_res.paste(im_cur, (i * w, 0))

    plt.figure(figsize = (16, 4))
    plt.imshow(im_res, cmap = 'binary')
    plt.axis('off')
    plt.show()

    if savepath != None:
        im_res.save(savepath)

if __name__ == "__main__":
    import doctest
    doctest.testmod()