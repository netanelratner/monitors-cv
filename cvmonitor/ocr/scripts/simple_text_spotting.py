import cv2
import numpy as np
from cvmonitor.ocr.text_spotting import text_spotting
from cvmonitor.image_align import align_by_4_corners


def read_annotation_file(file_path):

    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    ann = {}
    for line in lines:

        words = line.split(',')
        words = [word.strip() for word in words if len(word.strip()) > 0]

        n = 0
        N = len(words)
        while n < N: # iterate over all words in current line

            if n == 0:

                current_dict = {}
                key = words[n]
                ann[key] = current_dict
                n += 1

            else:

                if key == 'screen':
                    # array of 4 coordinates (x,y)
                    corners = [words[n + 1], words[n + 2], words[n + 3], words[n + 4],
                               words[n + 5], words[n + 6], words[n + 7], words[n + 8]]
                    corners = [int(float(corner)) for corner in corners] # cast to int
                    current_dict['corners'] = np.array(corners).astype(np.int)
                    n += 9

                else:

                    word = words[n]

                    if word =='val':
                        # scalar value (int or float)
                        val = words[n + 1]

                        if val.isnumeric():
                            # val is a number - check if int or float
                            if is_int(val): # val is int
                                val = int(val)
                            else: # val is float
                                val = float(val)
                        # else - val is str

                        current_dict['val'] = np.array(val, dtype=np.float)
                        n += 2

                    elif word == 'Position': # dtype is int
                        # array of [left, top, width, height] or maybe [top, left , width, height]
                        current_dict['box'] = np.array([words[n+1], words[n+2], words[n+3], words[n+4]], dtype=np.int)
                        n += 5

    return ann


def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


def enlarge_box(box, percent=0.2):
    """
    box should be in ltrb format: [left, top, right, bottom]
    """

    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]

    width = right - left
    height = bottom - top

    boundary_x = int(percent * width)
    boundary_y = int(percent * height)

    left -= boundary_x
    right += boundary_x
    top -= boundary_y
    bottom += boundary_y

    box_out = [left, top, right, bottom]

    return box_out


def change_box_type(box_in, type_in, type_out='ltrb'):

    """
    types:
        ltrb: [left, top, right, bottom]
        ltwh: [left, top, width, height]
        tlwh: [top, left, width, height]

    """

    if type_in == 'ltrb':

        left = box_in[0]
        top = box_in[1]
        right = box_in[2]
        bottom = box_in[3]

    elif type_in == 'ltwh':

        left = box_in[0]
        top = box_in[1]
        width = box_in[2]
        height = box_in[3]

        right = left + width
        bottom = top + height

    elif type_in == 'tlwh':

        left = box_in[1]
        top = box_in[0]
        width = box_in[2]
        height = box_in[3]

        right = left + width
        bottom = top + height

    if type_out == 'ltrb':
        box_out = [left, top, right, bottom]
    elif type_out == 'ltbr':
        box_out = [left, top, bottom, right]

    return box_out


def change_corners_type(corners, type_in, type_out='xyxy'):

    """
    types:
        xxyy: [tl_x, tr_x, br_x, bl_x, tl_y, tr_y, br_y, bl_y]
        xyxy: [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
    """

    if type_in == 'xyxy':

        tl_x = corners[0]
        tl_y = corners[1]
        tr_x = corners[2]
        tr_y = corners[3]
        br_x = corners[4]
        br_y = corners[5]
        bl_x = corners[6]
        bl_y = corners[7]

    elif type_in == 'xxyy':

        tl_x = corners[0]
        tl_y = corners[4]
        tr_x = corners[1]
        tr_y = corners[5]
        br_x = corners[2]
        br_y = corners[6]
        bl_x = corners[3]
        bl_y = corners[7]


    if type_out == 'xyxy':
        corners_out = np.array([[tl_x, tl_y],
                                [tr_x, tr_y],
                                [br_x, br_y],
                                [bl_x, bl_y]])
    # elif type_out == 'xxyy':
    #     corners_out = [tl_x, tr_x, br_x, bl_x, tl_y, tr_y, br_y, bl_y]

    return corners_out


def process_annotation_dict(ann_dict):

    expected_boxes = []

    for key, val in ann_dict.items():

        bbox = change_box_type(val['box'], type_in='ltwh', type_out='ltrb')

        bbox = enlarge_box(bbox, percent=0.2)

        expected_box = {'field': key,
                        'bbox': bbox,
                        'true_val': val['val']}

        expected_boxes.append(expected_box)

    return expected_boxes

if __name__ == '__main__':

    ann_file = 'cvmonitor/test/data/BneiZIon4_1.txt'

    img_path = 'cvmonitor/test/data/11.jpg'
    img_path = 'cvmonitor/test/data/BneiZIon4_1.tiff'

    predict_on_warped = False

    # model parameters
    visualize = True
    prob_threshold = 0.5
    max_seq_len = 6
    iou_threshold = 0.5
    model_type = 'FP32'  # 'FP16' # 'FP32'
    rgb2bgr = False # if True, channels order will be reversed

    # load image
    img = cv2.imread(img_path, -1)

    # read annotations
    ann = read_annotation_file(ann_file)

    # get screen corners
    ann_screen = ann.pop('screen', None)
    if ann_screen is not None:

        # get annotation corners
        corners_ann = ann_screen['corners']

        # change corners type
        corners = change_corners_type(corners_ann, type_in='xxyy', type_out='xyxy')

        # align img and annotions by corners
        img_warped, M = align_by_4_corners(img, corners, shape_out=(1280,768), margin_percent=0.1)

        # note that bounding boxes need not be aligned, since they will be taken from the aligned image

        # cv2.imshow('image', img)
        # cv2.imshow('warped', img_warped)
        # cv2.waitKey(0)


    # load model
    model = text_spotting.Model(visualize=visualize, prob_threshold=prob_threshold, max_seq_len=max_seq_len, iou_threshold=iou_threshold, model_type=model_type, rgb2bgr=rgb2bgr)

    # get expected boxes
    expected_boxes = process_annotation_dict(ann)
    # expected_boxes = None

    # predict text
    texts, boxes, scores, frame = model.forward(img, expected_boxes=expected_boxes)

    cv2.imshow('Results', frame)

    if predict_on_warped:
        texts, boxes, scores, frame = model.forward(img_warped, expected_boxes=expected_boxes)
        cv2.imshow('Results Warped', frame)

    cv2.waitKey(0)

    print('Done!')
