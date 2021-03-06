import os
import numpy as np
import cv2
import natsort
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from cvmonitor.ocr.corners_tracker import track_corners
from cvmonitor.image_align import align_by_4_corners

DISPLAY = True
FPS = 30.0

## Choose scenario
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion2/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion3/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion5_1/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/BneiZion6/'
# scenarioPath = '/media/dalya/data/DataSets/CV4Corona/Dataset300320_0600/Rambam1/'
scenarioPath = 'cvmonitor/test/data'
scenarioPath = '/home/moshes2/datasets/monitors/BneiZion4'
scenarioPath = '/home/moshes2/datasets/monitors/Rambam3'

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

        img_aligned, M = align_by_4_corners(img, points, new_image_size=None, margin_percent=0)
        plt.figure(2)
        plt.imshow(img_aligned)
        # plt.scatter(corners_warped[:,0], corners_warped[:,1], c='r')
        plt.show(block=False)

    points = track_corners(img, timeStamp, points)

    if DISPLAY:
        plt.figure(1)
        plt.cla()
        plt.imshow(img)
        plt.scatter(points[:,0], points[:,1], c='r')
        plt.show(block=False)

        img_aligned, M = align_by_4_corners(img, points, new_image_size=None, margin_percent=0)
        plt.figure(2)
        plt.cla()
        plt.imshow(img_aligned)
        # plt.scatter(corners_warped[:,0], corners_warped[:,1], c='r')
        plt.show(block=False)
        plt.pause(0.1)


        aaa=1