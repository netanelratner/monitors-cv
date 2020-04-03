import string
import argparse
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import cv2
import requests
import logging
import hashlib
 
from tqdm import tqdm
from .model import Model
from .import get_models
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



SEGMENTS_TYPES = {

     # ivac
     "Medication Name": False,
     "Volume Left to Infuse": True,
     "Volume to Insert": True,
     "Infusion Rate": True,

    # respirator
     "Ventilation Mode": False,
     "Tidal Volume": True,
     "Expiratory Tidal Volume": True,
     "Rate": True,
     "Total Rate": True,
     "Peep": True,
     "Ppeek": True,
     "FIO2": True,
     "Arterial Line": True,
     "I:E Ratio": True,
     "Inspiratory Time": True,

    # monitor
    "Heart Rate": True,
    "SpO2": True,
    "RR": True,
    "IBP-Systole": True,
    "IBP-Diastole": True,
    "NIBP-Systole": True,
    "NIBP-Diastole": True,
    "Temp": True,
    "etCO2": True,

}


def set_parameters():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='demo_image_monitor', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='cvmonitor/ocr/PreTrained/TPS-ResNet-BiLSTM-Attn.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=5, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt, rest = parser.parse_known_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if opt.rgb:
        opt.input_channel = 3

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    return opt


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        self.numerics = []
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i
            if char.isnumeric():
                self.numerics.append(i)

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        # for index, l in enumerate(length):
        #     text = ''.join([self.character[i] for i in text_index[index,:]])
        #     texts.append(text)
        # return texts

        for text_raw in text_index:
            text = ''.join([self.character[i] for i in text_raw])
            texts.append(text)
        return texts


class ModelOCR(object):

    def __init__(self):

        # set parameters
        self.opt = set_parameters()

        # initialize text-label and text-index converter
        self.converter = AttnLabelConverter(self.opt.character)


        self.opt.num_class = len(self.converter.character)


    def load_model(self, path):

        opt = self.opt

        # load architecture
        model = Model(opt)

        # move to device
        model = torch.nn.DataParallel(model).to(device)

        # load weights
        model.load_state_dict(torch.load(path, map_location=device))

        # set model mode to eval
        model.eval()

        self.model = model


    def preprocess_inputs(self, bbox_list, image):

        # crop bboxes from image
        images_list_raw = []
        for bb in bbox_list:
            images_list_raw.append(image[bb[1]:bb[3], bb[0]:bb[2]])

        images_list = []
        for single_roi in images_list_raw:
            images_list.append(cv2.resize(single_roi, (100, 32)))

        self.images_list = images_list

        return images_list


    def predict(self, images_list):

        opt = self.opt
        converter = self.converter
        model = self.model

        with torch.no_grad():

            batch_size = len(images_list)
            stackedImages = np.stack(images_list)
            stackedImages = np.expand_dims(stackedImages, axis=1)
            stackedImages = stackedImages / 255.
            torchStackedImages = torch.from_numpy(stackedImages[..., 0]) # take only first channel since
            torchStackedImages = torchStackedImages.type(torch.float32)
            torchStackedImages = torchStackedImages.sub_(0.5).div_(0.5)
            torchStackedImages = torchStackedImages.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            # predict
            preds = model(torchStackedImages, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)

            # convert indices to text
            preds_str = converter.decode(preds_index, length_for_pred)

            self.preds = preds
            self.preds_str = preds_str

            return preds, preds_str

    # def log_predictions(self, verbose=0):
    #
    #     log_file = self.log_file
    #     preds = self.preds
    #     preds_str = self.preds_str
    #
    #     log = open(log_file, 'a')
    #     dashed_line = '-' * 80
    #     head = f'{"predicted_labels":25s}\tconfidence score'
    #
    #     log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')


    def display(self, image_np, bbox_list, texts, scores, verbose=0):
        threshold = float(os.environ.get('CVMONITOR_THRESHOLD_CHARACTER',"0.9"))
        # preds = self.preds
        # preds_str = self.preds_str

        if verbose > 0:
            dashed_line = '-' * 80
            head = f'{"predicted_labels":25s}\tconfidence score'
            print(f'{dashed_line}\n{head}\n{dashed_line}')

        imm = image_np.copy()

        madadim=[]
        for text, score, rec in zip(texts, scores, bbox_list):
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.8
            # Blue color in BGR
            color = (0, 255, 0)
            imm =cv2.rectangle(imm,(rec[0],rec[1]),(rec[2],rec[3]),(0,255,0),1)
            imm = cv2.putText(imm, text, (rec[0],rec[1]-5), font,
                                fontScale, color, 2, cv2.LINE_AA)
            if verbose > 0:
                print(f'{text:25s}\t{score:0.4f}')

        return imm

    def ocr(self, expected_boxes, image, threshold, save_image_path=None):
        bbox_list = []
        are_numeric = []
        for box in expected_boxes:
            bbox_list.append(box['bbox'])
            if box['name'] not in SEGMENTS_TYPES:
                are_numeric.append(False)
            else:
                are_numeric.append(SEGMENTS_TYPES[box['name']])
        texts, preds = self.detect(bbox_list, image, are_numeric, save_image_path)
        more_texts = []
        for t, p, b in zip(texts,preds,bbox_list):
            if p>threshold:
                more_texts.append(texts)
            else:
                more_texts.append(None)
        return texts, preds

    def detect(self, bbox_list, image, is_numeric=[], save_image_path=None):
        """
        # bbox_list is [[left,top,right,bottom],...]
        """

        # ------------------
        # Preprocessing
        # ------------------

        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # FIXME: check if RGB or BGR

        # histogram equalization
        clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(16,16))
        image = clahe.apply(image)

        # simple streching
        # image -= image.min()
        # image = (image / image.max() * 255.).astype(np.uint8)

        image = np.stack([image, image, image],axis=-1)

        # pre-process input
        images_list = self.preprocess_inputs(bbox_list=bbox_list, image=image)

        # ------------------

        # predict
        preds, preds_str = self.predict(images_list)


        threshold_character = float(os.environ.get('CVMONITOR_THRESHOLD_CHARACTER',"0.9"))
        threshold_numeric = float(os.environ.get('CVMONITOR_THRESHOLD_CHARACTER',"0.95"))
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        texts=[]
        scores = []
        n = -1 # counter
        for pred, pred_prob, numeric in zip(preds_str, preds_prob, is_numeric):
            n += 1
            if not numeric:
                pred_max_prob, _ = pred_prob.max(dim=1)
                threshold = threshold_character

                text = ''
                score = 1
                pred_text = pred.replace('[s]', '[')
                for c, p in zip(pred_text, pred_max_prob):
                    if p > threshold and c != '[':
                        text += c
                        score *= p
                if len(text) == 0:
                    score = 0.0

            else:
                # pred_max_prob, _ = pred_prob[:,self.converter.numerics].max(dim=1)
                # threshold = threshold_character

                # recalculate probabilities for numeric characters only
                pred_numeric = preds[n, :, self.converter.numerics]
                pred_prob_numeric = F.softmax(pred_numeric, dim=1)

                pred_max_prob, preds_index = pred_prob_numeric.max(dim=1)

                preds_index += self.converter.numerics[0]

                # convert indices to text
                pred = ''.join([self.converter.character[i] for i in preds_index])

                # preds_str = self.converter.decode(preds_index, length=1)

                threshold = threshold_numeric

                text = ''
                score = 1
                pred_text = pred.replace('[s]', '[')
                for c, p in zip(pred_text, pred_max_prob):
                    if p > threshold and c != '[':
                        text += c
                        score *= p
                if len(text) == 0:
                    score = 0.0

            texts.append(text)
            scores.append(score)
        # display
        if save_image_path is not None:

            # add predicted text to image
            res = self.display(image, bbox_list, texts, scores, verbose=1)

            cv2.imwrite(save_image_path, res)
            
        return texts, scores




def build_model():
    path = get_models()['TPS-ResNet-BiLSTM-Attn.pth']
    model_ocr = ModelOCR()
    model_ocr.load_model(path)
    return model_ocr


if __name__ == '__main__':


    # inputs
    bbox_list = np.load('demo_image_monitor/11_recs.npy')
    image = cv2.imread('demo_image_monitor/11.jpg')

    save_image_path = 'demo_image_monitor_out/res.jpg' # None

    # main
    # bbox_list is [[left,top,right,bottom]]
    res = detect(bbox_list=bbox_list, image=image, save_image_path=save_image_path)

    print('Done!')

