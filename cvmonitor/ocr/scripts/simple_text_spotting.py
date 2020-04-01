import cv2
from cvmonitor.ocr.text_spotting import text_spotting





if __name__ == '__main__':

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
