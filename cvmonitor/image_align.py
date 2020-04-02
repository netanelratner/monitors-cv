from flask import Flask, Blueprint, request, abort, Response
import math
import cv2
import imageio
import ujson as json
import numpy as np
from pyzbar import pyzbar
import qrcode
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from .ocr import monitor_ocr
from prometheus_client import Summary
import pytesseract
from pylab import imshow, show
import exifread
import logging
from .qr import find_qrcode
np.set_printoptions(precision=3)



def order_points(pts):
    """
    order such that [top-left,top-right,bottom-right,bottom-left]
    """
    # find which points are the topmost points
    by_top = np.argsort(pts[:,1].copy())
    by_left = np.argsort(pts[:,0].copy())
    top_left = [int(x) for x in by_top[:2] if x in by_left[:2]][0]
    bottom_left = [int(x) for x in by_left if x != top_left][0]
    top_right = [int(x) for x in by_top if x != top_left][0]
    bottom_right = [int(x) for x in [0,1, 2,3,] if x not in [top_left, bottom_left , top_right]][0]
    return pts[[top_left, top_right, bottom_right, bottom_left],:]


def rotmat(deg,w,h):
    r = math.radians(deg)
    return np.array([
        [ math.cos(r), -math.sin(r), h],
        [ math.sin(r), math.cos(r), w]
    ],dtype=np.float32)

# https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c

def get_rotation_transform(height, width, angle):
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    return rotation_mat

def get_exif_rotation(im_file):
    rotation = None
    # Try with exif:
    tags = exifread.process_file(im_file)
    if isinstance(tags,dict) and 'Image Orientation' in tags:
        orientation = tags['Image Orientation'].values[0]
        if orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        elif orientation == 3:
            rotation = 180
        else:
            rotation = 0
    im_file.seek(0)
    return im_file, rotation


def get_qr_rotation(image, detected_qrcode=None, qrprefix=''):
    rotation = None
    if not detected_qrcode:
        detected_qrcode = find_qrcode(image, qrprefix)
    if detected_qrcode:
        src_pts= np.array([(p.x,p.y) for p in detected_qrcode.polygon],np.float32)
        width = image.shape[1]
        height = image.shape[0]
        
        y_mean = np.mean(src_pts[:,1])
        flip_y = y_mean > height//2
        
        x_mean = np.mean(src_pts[:,0])
        flip_x = x_mean > width//2

        if flip_x and flip_y:
            rotation = 180

        if flip_y and not flip_x:
            rotation = 90

        if flip_x and not flip_y:
            rotation = -90
    return rotation, detected_qrcode


def rotate_image(image, rotation):
    logging.debug(f'Image is {rotation}')
    if rotation == -90:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image

    return image



def get_oriented_image(im_file, use_exif=True, use_qr=False, detected_qrcode=None, qrprefix=''):
    """
    Orient an image by it's exif data or by qr code that is expected to be on
    the image top-left side.
    :return: the rotated image, qrcode *in original cooordinates*, rotation in angles
    """

    # try exif
    im_file, rotation  = get_exif_rotation(im_file)

    image = imageio.imread(im_file)

    # if no oritenation in exif or don't use exif, maybe try qr code:
    detected_qrcode = None
    if rotation is None and use_qr or not use_exif:
        rotation, detected_qrcode = get_qr_rotation(image, detected_qrcode, qrprefix)

    image = rotate_image(image, rotation)

    # don't waste the qr code detection
    return image, detected_qrcode, rotation


def align_by_qrcode(image : np.ndarray, detected_qrcode, qrsize=100, boundery = 50.0, qrprefix=''):
    """
    Aling image by qrcode, normalize by qrcode size.
    Assumption - the image orietnation is already correct, if there is a qr code it's on the top
    left side of the image.
    WARNING: Aligining an image by a single qr code is pretty unstable. be carefull with this.
    """


    src_pts= np.array([(p.x,p.y) for p in detected_qrcode.polygon],np.float32)
    width = image.shape[1]
    height = image.shape[0]


    src_pts=order_points(src_pts).astype(np.float32)
    
    tgt_pts = np.array([[boundery,boundery],[qrsize+boundery,boundery],[qrsize+boundery,qrsize+boundery],[boundery,qrsize+boundery]],np.float32)
    shape_pts = np.array([[0,0],[width,0],[width,height],[0.0,height]],np.float32)
    M = cv2.getPerspectiveTransform(src_pts, tgt_pts)
    res = M @ np.concatenate([shape_pts,np.ones((4,1))],1).transpose()
    for r in range(4):
        res[:,r]/=res[-1,r]
    width = int(np.ceil(max(res[0,:]))) #+ int(np.floor(min(res[0,:])))
    height = int(np.ceil(max(res[1,:]))) #+ int(np.floor(min(res[1,:])))
    warped = cv2.warpPerspective(image, M,(width,height))
    return warped, M


def align_by_4_corners(image, corners, shape_out=(1280,768), margin_percent=0.1):
    """
    Rescale an Image given its corners.
    outer_margin is the margin outside these points
    """


    # verify corners order
    src_pts=order_points(corners).astype(np.float32)

    # calculate wanted screen corners in warped image

    height_screen = shape_out[0]
    width_screen = shape_out[1]

    margin_x = int(width_screen * margin_percent)
    margin_y = int(height_screen* margin_percent)

    tgt_pts = np.float32([[margin_x, margin_y], # tl
                          [width_screen-margin_x, margin_y], # tr
                          [width_screen-margin_x, height_screen-margin_y], # br
                          [margin_x, height_screen-margin_y]])# bl


    # calculate perspective transformation
    M = cv2.getPerspectiveTransform(src_pts, tgt_pts)


    # calculate transformed image corners
    height_in = image.shape[0]
    width_in = image.shape[1]

    #shape_pts - size of image at the end
    shape_pts = np.array([[0, 0], [width_in, 0], [width_in, height_in], [0.0, height_in]], np.float32)

    # warp image corners
    res = M @ np.concatenate([shape_pts, np.ones((4, 1))], 1).transpose()
    for r in range(4):
        res[:, r] /= res[-1, r]
    width = int(np.ceil(max(res[0, :])))  # + int(np.floor(min(res[0,:])))
    height = int(np.ceil(max(res[1, :])))  # + int(np.floor(min(res[1,:])))
    warped = cv2.warpPerspective(image, M, (width, height))
    # warped = warped[1:int(verti_length)+1,1:int(horiz_length)+1]

    return warped, M


# def align_by_4_corners(image, corners, margin_percent=0):
#     """
#     Rescale an Image given its corners.
#     outer_margin is the margin outside these points
#     """
#     width = image.shape[1]
#     height = image.shape[0]
#     margin_x = width * margin_percent /100
#     margin_y = height* margin_percent /100
#
#     src_pts = order_points(corners).astype(np.float32)
#
#     #src_pts = monitor corners - margins
#     # src_pts = np.float32([[corners[0], corners[1]],
#     #                       [corners[2], corners[3]],
#     #                       [corners[4], corners[5]],
#     #                       [corners[6], corners[7]]])
#
#     #tgt_pts = size of monitor + boundaries
#     horiz_length = np.sqrt(pow((corners[0]-corners[2]),2)+pow((corners[1]-corners[3]),2)) + 2*margin_x
#     verti_length = np.sqrt(pow((corners[0]-corners[6]),2)+pow((corners[1]-corners[7]),2)) + 2*margin_y
#
#     tgt_pts=np.float32([[0, 0],
#                        [horiz_length, 0],
#                        [horiz_length, verti_length],
#                        [0, verti_length]])
#
#     #shape_pts - size of image at the end
#     shape_pts = np.array([[0, 0], [width, 0], [width, height], [0.0, height]], np.float32)
#
#     M = cv2.getPerspectiveTransform(src_pts, tgt_pts)
#     res = M @ np.concatenate([shape_pts, np.ones((4, 1))], 1).transpose()
#     for r in range(4):
#         res[:, r] /= res[-1, r]
#     width = int(np.ceil(max(res[0, :])))  # + int(np.floor(min(res[0,:])))
#     height = int(np.ceil(max(res[1, :])))  # + int(np.floor(min(res[1,:])))
#     warped = cv2.warpPerspective(image, M, (width, height))
#     warped = warped[1:int(verti_length)+1,1:int(horiz_length)+1]
#     return warped, M


