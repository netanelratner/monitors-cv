import cv2
import numpy as np
from cvmonitor.ocr.text_spotting import text_spotting


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

                if key == 'monitor':
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

if __name__ == '__main__':

    ann_file = 'cvmonitor/test/data/BneiZIon4_1.txt'
    ann = read_annotation_file(ann_file)

    img_path = 'cvmonitor/test/data/11.jpg'
    img_path = 'cvmonitor/test/data/BneiZIon4_1.tiff'

    visualize = True
    prob_threshold = 0.5
    max_seq_len = 6
    iou_threshold = 0.5
    model_type = 'FP32'  # 'FP16' # 'FP32'

    model = text_spotting.Model(visualize=visualize, prob_threshold=prob_threshold, max_seq_len=max_seq_len, iou_threshold=iou_threshold, model_type=model_type)

    img = cv2.imread(img_path, -1)

    expected_boxes = None

    texts, boxes, scores, frame = model.forward(img, expected_boxes=expected_boxes)

    cv2.imshow('Results', frame)

    cv2.waitKey(0)

    print('Done!')
