
# functions for resizing and scaling input data.
#

import cv2
import numpy as np

def resize_pad(im: np.array, target_size: int) -> np.array:
    """from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#using-opencv.

    Parameters
    ----------
    im : np.array
        Description of parameter `img`.
    target_size : int
        Description of parameter `new_size`.

    Returns
    -------
    np.array
        Description of returned object.

    """

    old_size = im.shape[:2] # old_size is in (height, width) format
    # computes
    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im
