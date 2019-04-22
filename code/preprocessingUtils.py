import numpy as np

def normImg(img):
    normImg = (img - np.mean(img))/np.std(img)
    return normImg