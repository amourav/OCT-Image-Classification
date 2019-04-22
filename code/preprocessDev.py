#%%
import numpy as np
import skimage

import os
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

testPath = r"D:\Projects\OCT-Image-Classification\RawData\OCT2017\test\CNV\CNV-53018-1.jpeg"
testOutPath = r"D:\Projects\OCT-Image-Classification\RawData\OCT2017\test\CNV\testImg.npy"


def normImg(img):
    normImg = (img - np.mean(img))/np.std(img)
    return normImg

testImg = Image.open(testPath)
newSize = (224, 224)

testImgReSized = testImg.resize(newSize, Image.ANTIALIAS)
imgArr = np.asarray(testImgReSized)
imgNorm = normImg(imgArr)

#%

