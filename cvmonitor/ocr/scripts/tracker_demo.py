import os
import numpy as np
import cv2
import natsort
import matplotlib.pyplot as plt
from cvmonitor.ocr.corners_tracker import track_corners

DISPLAY = True
FPS = 30.0

## Choose scenario
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion2/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion3/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion5_1/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion6/'
scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/Rambam1/'

# Find all images and sort them
imagesList = os.listdir(scenarioPath)
imagesList = natsort.natsorted(imagesList)

# Run main loop
for ind, imgName in enumerate(imagesList):
    timeStamp = ind/FPS

    imgNameFull = os.path.join(scenarioPath, imgName)
    img = cv2.imread(imgNameFull, -1)

    if ind == 0:  # initialize points
        plt.figure(1)
        plt.imshow(img)
        points = np.float32(plt.ginput(4))
        plt.close(1)

    points = track_corners(img, timeStamp, points)

    if DISPLAY:
        plt.figure(1)
        plt.cla()
        plt.imshow(img)
        plt.scatter(points[:,0], points[:,1], c='r')
        plt.show()
        aaa=1