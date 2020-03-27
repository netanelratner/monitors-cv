#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import io
import imageio
from uuid import uuid4
import qrcode
import random
import numpy as np
import requests
import time
import pickle
import os
import copy
from pyzbar import pyzbar
from PIL import Image, ImageDraw, ImageFont
from cvmonitor.ocr import monitor_ocr
import pytesseract
import datetime
import argparse
QRSIZE=100

SEND_TO_SERVER = True
name_list = [
"בנימין נתניהו",
 "יולי אדלשטיין",
 'ישראל כ"ץ',
 "גלעד ארדן",
 "גדעון סער",
 "מירי רגב",
 "יריב לוין",
 "יואב גלנט",
 "ניר ברקת",
 "גילה גמליאל",
 "אבי דיכטר",
 "זאב אלקין",
 "חיים כץ",
 "אלי כהן",
 "צחי הנגבי",
 "אופיר אקונ0יס",
 "יובל שטייניץ",
 "ציפי חוטובלי",
 "דודי אמסלם",
 "גדי יברקן",
 "אמיר אוחנה",
 "אופיר כץ",
 "אתי עטייה",
 "יואב קיש",
 "דוד ביטן",
 "קרן ברק",
 "שלמה קרעי",
 "מיקי זוהר",
 "יפעת שאשא-ביטון",
 "שרן השכל",
 "מיכל שיר",
 "קטי שטרית",
 "פטין מולא",
 "מאי גולן",
 "טלי פלוסקוב",
 "עוזי דיי",
 "בני גנץ",
 "יאיר לפיד",
 "משה יעלון",
 "גבי אשכנזי",
 "אבי ניסנקורן",
 "מאיר כהן",
 "מיקי חיימוביץ'",
 "עפר שלח",
 "יועז הנדל",
 "אורנה ברביבאי",
 "מיכאל ביטון",
 "חילי טרופר",
 "צבי האוזר",
 "אורית פרקש-הכהן",
 "קארין אלהרר",
 "מירב כהן",
 "יואל רזבוזוב",
 "אסף זמיר",
 "יזהר שי",
 "אלעזר שטרן",
 "מיקי לוי",
 "עומר ינקלביץ'",
 "פנינה תמנו-שטה",
 "ר'דיר מריח",
 "רם בן ברק",
 "אלון שוסטר",
 "יואב סגלוביץ'",
 "רם שפע",
 "בועז טופורובסקי",
 "אורלי פרומן",
 "איתן גינזבורג",
 "אנדריי קוז'ינוב",
 "עידן רול",
 "איימן עודה",
 "מטאנס שחאדה",
 "אחמד טיבי",
 "מנסור עבאס",
 "עאידה תומא סלימאן",
 "וליד טאהא",
 "עופר כסיף",
 "היבא יזבק",
 "אוסאמה סעדי",
 "יוסף ג'בארין",
 "סעיד אלחרומי",
 "ג'אבר עסאקלה",
 "סאמי אבו שחאדה",
 "סונדוס סאלח",
 "אימאן ח'טיב יאסי",
 "אריה דרעי",
 "יצחק כהן",
 "משולם נהרי",
 "יעקב מרגי",
 "יואב בן צור",
 "מיכאל מלכיאלי",
 "משה ארבל",
 "ינון אזולאי",
 "משה אבוטבו",
 "אביגדור ליברמן",
 "עודד פורר",
 "יבגני סובה",
 "אלי אבידר",
 "יוליה מלינובסקי",
 "חמד עמאר",
 "אלכס קושני",
 "יעקב ליצמן",
 "משה גפני",
 "מאיר פרוש",
 "אורי מקלב",
 "יעקב טסלר",
 "יעקב אשר",
 "ישראל אייכל",
 "עמיר פרץ",
 "ניצן הורוביץ",
 "תמר זנדברג",
 "איציק שמולי",
 "מרב מיכאלי",
 "יאיר גול",
 "נפתלי בנט",
 "רפי פרץ",
 "איילת שקד",
 "בצלאל סמוטריץ'",
 "מתן כהנא",
 "אופיר סופ",
 "אורלי לוי-אבקסיס"
]

monitor = {
     "Heart Rate": lambda: random.randint(45, 120),
     "SpO2": lambda: random.randint(10, 100),
     "RR": lambda: random.randint(10, 100),
     "IBP": lambda: random.randint(10, 100),
     "NIBP": lambda: random.randint(10, 100),
     "Temp": lambda: random.randint(10, 100),
     "etC02": lambda: random.randint(10, 100),
}

respirator = {
     "Ventilation Mode": lambda: random.choice(["ABC", "DEF"]),
     "Tidal Volume": lambda: random.randint(350, 600),
     "Expiratory Tidal Volume": lambda: random.randint(10, 100),
     "Rate": lambda: random.randint(10, 100),
     "Total Rate": lambda: random.randint(10, 100),
     "Peep": lambda: random.randint(10, 100),
     "Ppeek": lambda: random.randint(10, 100),
     "FIO2": lambda: random.randint(10, 100),
     "Arterial Line": lambda: random.randint(10, 100),
     "I:E Ratio": lambda: random.randint(10, 100),
     "Inspiratory Time": lambda: random.randint(10, 100),
}

ivac = {
     "Medication Name": lambda: random.choice(['MEDI', "WINE", "COFFE", "BEER"]),
     "Volume Left to Infuse": lambda: random.randint(10, 13),
     "Volume to Insert": lambda: random.randint(10, 13),
     "Infusion Rate": lambda: random.randint(10, 13),
}

devices = {
     "ivac": ivac,
     "respirator": respirator,
     "monitor": monitor
}


def get_qr_code(title):
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
    text = f'cvmonitors-{title}-{uuid4().hex[:16]}'
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    qrsize = random.randint(120,180)
    img = cv2.resize(np.array(img,dtype=np.uint8)*255,(qrsize,qrsize))
    img= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    return img, text


def create_segments(device_type, fontScale, thickness, image_size=[1000, 1200], x_start=200, y_start=200):
    size=np.array(cv2.getTextSize(text=str('COFFE'), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,thickness=thickness)[0])+10
    y_step = size[1]+50
    x_step = size[0]+50
    x = x_start
    y = y_start
    segments = []
    for m in devices[device_type]:
        if x+x_step >= image_size[1]-50:
            x = x_start
            y += y_step
        segments.append({
            "top": max(y-20,0),
            "left": max(x-20,0),
            "bottom": int(min(y+size[1],image_size[0]-10)),
            "right": int(min(x+size[0],image_size[1]-10)),
            "name": m,
            
        })
        if x+x_step < image_size[1]-50:
            x += x_step
    return segments


def fill_segments(segments, device_type):
    device = devices[device_type]
    values = []
    colors = []
    for s in segments:
        values.append({"segment_name": s['name'], "value": device[s['name']]()})
        colors.append((random.randint(20,255),random.randint(20,255),random.randint(20,255)))
    return values, colors


def change_values(values):
    for i in range(len(values)):
        if str(values[i]['value']).isnumeric():
            values[i]['value'] += random.randint(-3, 3)
    return values

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def generate_picture(qrcode, image_size, segments, values, colors, fontScale,thickness):
    image = np.zeros((image_size[0],image_size[1],3),dtype=np.uint8)+40
    image[12:qrcode.shape[0]+12, 18:qrcode.shape[1]+18, :] = qrcode
    for s, v, color in zip(segments, values,colors):
        size=cv2.getTextSize(text=str(v['value']), fontFace=cv2.FONT_HERSHEY_PLAIN, 
            fontScale=fontScale,thickness=thickness)
        image = cv2.putText(img=image, text=str(v['value']), org=(s['left'], s['top']+size[0][1]+20), fontFace=cv2.FONT_HERSHEY_PLAIN, 
            fontScale=fontScale, color=color, thickness=thickness)
    return image

class Device():

    def __init__(self, device_type, patient, room_number):
        self.index = 0
        self.monitor_id = None
        self.device_type = device_type
        self.image_size = [1000, 1200]
        self.fontScale = random.randint(3, 5)
        self.thickness= random.randint(3, 8)
        self.segments = create_segments(device_type,  self.fontScale, self.thickness, self.image_size)
        self.draw_segments = copy.deepcopy(self.segments)
        self.values, self.colors = fill_segments(self.segments, device_type)
        qrcode, self.qrtext = get_qr_code(device_type)
        self.qrcode =  rotate_image(qrcode, float(random.randint(-10, 10)))
        self.patient = patient
        self.room_number = room_number
        
        

    def picture(self):
        return generate_picture(
            self.qrcode, self.image_size, self.draw_segments, self.values,self.colors,
            self.fontScale,self.thickness,
            )

    def change_values(self):
        self.values = change_values(self.values)


def fill_rooms(device_count):
    active_devices = []
    for i in range(device_count):
        room = random.randint(1, 400)
        patient = random.choice(name_list)
        active_devices.append(Device('ivac', patient, room))
        active_devices.append(Device('monitor', patient, room))
        active_devices.append(Device('respirator', patient, room))
    return active_devices

def order_points(pts):
    """
    order such that [top-left,top-right,bottom-right,bottom-left]
    """
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_qrcode(image, prefix):
    if len(image.shape)==3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    image = clahe.apply(image)
    decodedObjects = pyzbar.decode(image)
    detected_qrcode = None
    for obj in decodedObjects:
        text = obj.data.decode()
        if text.startswith(prefix):
            detected_qrcode = obj
            break
    return detected_qrcode

def align_by_qrcode(image, detected_qrcode, qrsize=QRSIZE, boundery = 20.0):
    """
    Aling image by qrcode, normalize by qrcode size
    """
    tgt_pts = np.array([[boundery,boundery],[qrsize,boundery],[qrsize,qrsize],[boundery,qrsize]],np.float32)
    shape_pts = np.array([[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0.0,image.shape[0]]],np.float32)
    src_pts= np.array([(p.x,p.y) for p in detected_qrcode.polygon],np.float32)
    src_pts=order_points(src_pts)
    M = cv2.getPerspectiveTransform(src_pts, tgt_pts)
    res = M @ np.concatenate([shape_pts,np.ones((4,1))],1).transpose()
    for r in range(4):
        res[:,r]/=res[-1,r]
    width = int(np.ceil(max(res[0,:]))) #+ int(np.floor(min(res [0,:])))
    height = int(np.ceil(max(res[1,:]))) #+ int(np.floor(min(res[1,:])))
    warped = cv2.warpPerspective(image, M,(width,height))
    return warped, M

def update_segments(image,segments,qrprefix='cvmonitor'):
    detected_qrcode = find_qrcode(image, qrprefix)
    if detected_qrcode is None:
        raise RuntimeError("Could not detect QR")
    data = detected_qrcode.data.decode()
    wraped, M = align_by_qrcode(image, detected_qrcode, qrsize=QRSIZE, boundery = 20)
    segments = copy.deepcopy(segments)
    for i,s in enumerate(segments):
        V = np.array([
            [s['left'],s['top'],1],
            [s['left'],s['bottom'],1],
            [s['right'],s['top'],1],
            [s['right'],s['bottom'],1]
        ])
        U = M @ V.transpose()
        for r in range(4):
            U[:,r]/=U[-1,r]
        s['left'] = int(min(U[0,0:2]))
        s['right'] = min(int(max(U[0,2:])),wraped.shape[1])
        s['top'] = int(min(U[1,0],U[1,2]))
        s['bottom'] = min(int(max(U[1,1],U[1,3])),wraped.shape[0])
    return wraped, segments

def draw_segements(image, segments,colors):
    for s,c in zip(segments,colors):
        image =cv2.rectangle(image,(int(s['left']),int(s['top'])),(int(s['right']),int(s['bottom'])),c,1)
    return image

model_ocr = monitor_ocr.build_model()
            
def send_all_pictures(url, active_devices):
    device_indxes = list(range(len(active_devices)))
    random.shuffle(device_indxes)
    for di in device_indxes:
        device = active_devices[di]
        image = device.picture()
        print(device.values)
        image = draw_segements(image, device.draw_segments,device.colors)
        
        if SEND_TO_SERVER:
            b = io.BytesIO()
            imageio.imwrite(b, image, format='jpeg')
            imageio.imwrite(device.qrtext+f'.{device.index}.jpg', image, format='jpeg')
            b.seek(0)
            headers={'Content-Type':'image/jpeg','X-IMAGE-ID':str(device.index)}
            if device.monitor_id is not None:
                headers['X-MONITOR-ID']=str(device.monitor_id)
                headers['X-TIMESTEMP']=str(datetime.datetime.utcnow().isoformat())
            res = requests.post(url + '/monitor_image', data=b,headers=headers)
            res_data = res.json()
            print(res_data)
            if 'nextImageId' in res_data:
                device.index = res_data['nextImageId']
            if 'monitorId' in res_data:
                device.monitor_id = 'monitorId'
        else:
            detected_qrcode = find_qrcode(image, '')
            if detected_qrcode is None:
                raise RuntimeError("Could not detect QR")
            data = detected_qrcode.data.decode()
            image, M = align_by_qrcode(image, detected_qrcode, qrsize=QRSIZE, boundery = 20)
            segments = device.segments
            texts = model_ocr.ocr(segments, image, threshold=0.2, save_image_path=device.qrtext +'test.jpg')
            print(texts)



def add_devices(url, active_devices):

    for di in range(len(active_devices)):
        device = active_devices[di]
        device_json = {
            "monitorId": device.qrtext,
            "patientId": device.patient,
            "imageId":0,
            "roomId": device.room_number,
            "deviceCategory": device.device_type,
            "segments": device.segments
        }
        res=requests.post(url + f'/monitor/{device.qrtext}', json=device_json)
        print(res.text)




def main():
    if os.path.exists('devices.pkl'):
        active_devices = pickle.load(open('devices.pkl', 'rb'))
    else:
        active_devices = fill_rooms(5)
        pickle.dump(active_devices, open('devices.pkl','wb'))
    
    for d in active_devices:
        picture=d.picture()
        picture, new_segments =update_segments(picture,d.segments,qrprefix='cvmonitor')
        d.segments = new_segments
        picture=draw_segements(picture,new_segments,d.colors)
        try:
            cv2.imwrite(d.qrtext+'.jpg', picture)
        except:
            cv2.imwrite(d.qrtext+'.jpg', d.picture())
            print(d.qrtext)
            exit(-1)
    url = 'http://cvmonitors.westeurope.cloudapp.azure.com'
    url = 'http://52.157.71.156'
    send_all_pictures(url, active_devices)
    if not SEND_TO_SERVER:
        exit(0)
    add_devices(url, active_devices)
    while True:
        send_all_pictures(url, active_devices)
        for d in active_devices:
            d.change_values()
        time.sleep(1)


if __name__ == "__main__":
    parser =  argparse.ArgumentParser()
    parser.add_argument('--no_send',action='store_true',help='dont send to server just create images')
    args = parser.parse_args()
    if args.no_send:
        SEND_TO_SERVER=False
    random.seed(0)
    main()
