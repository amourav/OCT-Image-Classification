#%%
import numpy as np
import skimage

import os
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

testPath = r"D:\Projects\OCT-Image-Classification\PreprocessedData\individual images\train"
subdirs = os.listdir(testPath)

for subdir in subdirs:
    imgClassPath = os.path.join(testPath, subdir)
    imgFiles = os.listdir(imgClassPath)

imgFilesSubset = np.random.choice(imgFiles, 10)

imgList = []
for imFile in imgFilesSubset:
    imgPath = os.path.join(imgClassPath, imFile)
    img = np.load(imgPath)
    imgList.append(img)