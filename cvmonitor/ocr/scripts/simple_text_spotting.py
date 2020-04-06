import os
import cv2
import numpy as np
from cvmonitor.ocr.text_spotting import text_spotting
from cvmonitor.image_align import align_by_4_corners
from cvmonitor.ocr.utils import process_annotation_dict, read_annotation_file, change_corners_type


if __name__ == '__main__':

    ann_file = None
    img_path = 'cvmonitor/test/data/11.jpg'

    # ann_file = 'cvmonitor/test/data/BneiZIon4_1.txt'
    # img_path = 'cvmonitor/test/data/BneiZIon4_1.tiff'
    # ann_file = 'cvmonitor/test/data/IMG-20200405-WA0005.txt'
    # img_path = 'cvmonitor/test/data/IMG-20200405-WA0005.jpg'

    # ann_file = 'data/monitors/BneiZion2/out.txt'
    # img_path = 'data/monitors/BneiZion2/00000001.tiff'

    output_dir = 'cvmonitor/ocr/scripts/output'
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(img_path).split('.')[0]

    predict_on_warped = False
    display = False

    aligned_image_size = (1280, 768)  # (width, height)! and NOT (rows, cols)
    align_margin_percent = 10  # [%] valid values are 0-40

    # model parameters
    visualize = True
    prob_threshold = 0.5
    max_seq_len = 6
    iou_threshold = 0.01 # 0.4
    model_type = 'FP32'  # 'FP16' # 'FP32'
    rgb2bgr = False # if True, channels order will be reversed

    # load image
    img = cv2.imread(img_path, -1)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    # read annotations
    if ann_file is not None:

        ann = read_annotation_file(ann_file)

        # get screen corners
        ann_screen = ann.pop('screen', None)
        if ann_screen is not None:

            # get annotation corners
            corners_ann = ann_screen['corners']

            # change corners type
            corners = change_corners_type(corners_ann, type_in='xxyy', type_out='xyxy')

            # align img and annotions by corners
            # img_warped, M = align_by_4_corners(img, corners, shape_out=(1280,768), margin_percent=0.)
            img_warped, M = align_by_4_corners(img, corners, new_image_size=aligned_image_size, margin_percent=align_margin_percent)

            # note that bounding boxes need not be aligned, since they will be taken from the aligned image

            # cv2.imshow('image', img)
            # cv2.imshow('warped', img_warped)
            # cv2.waitKey(0)

        # get expected boxes
        expected_boxes = process_annotation_dict(ann)

    else:

        expected_boxes = None

    # load model
    model = text_spotting.Model(visualize=visualize, prob_threshold=prob_threshold, max_seq_len=max_seq_len, iou_threshold=iou_threshold, model_type=model_type, rgb2bgr=rgb2bgr)

    # predict text
    texts, boxes, scores, frame = model.forward(img, expected_boxes=expected_boxes)

    # save output
    out_name = os.path.join(output_dir, img_name + '.png')
    cv2.imwrite(out_name, frame)


    if display:
        cv2.imshow('Results', frame)

    if predict_on_warped:
        expected_boxes = None
        texts, boxes, scores, frame = model.forward(img_warped, expected_boxes=expected_boxes)

        # save output
        out_name = os.path.join(output_dir, img_name + '_warped_{}_{}_margin_{}.png'.format(aligned_image_size[0], aligned_image_size[1], align_margin_percent))
        cv2.imwrite(out_name, frame)

        if display:
            cv2.imshow('Results Warped', frame)

    if display:
        cv2.waitKey(0)

    print('Done!')
