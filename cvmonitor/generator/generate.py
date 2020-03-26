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
from PIL import Image, ImageDraw, ImageFont

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
     "Saturation": lambda: random.randint(10, 100),
     "RR": lambda: random.randint(10, 100),
     "IBP": lambda: random.randint(10, 100),
     "NIBP": lambda: random.randint(10, 100),
     "Temp": lambda: random.randint(10, 100),
     "ETC02": lambda: random.randint(10, 100),
}

respirator = {
     "Breathing Method": lambda: random.choice(["ABC", "DEF"]),
     "Tidal Volume": lambda: random.randint(350, 600),
     "Expiratory Tidal Volume": lambda: random.randint(10, 100),
     "Rate": lambda: random.randint(10, 100),
     "Total Rate": lambda: random.randint(10, 100),
     "Peep": lambda: random.randint(10, 100),
     "Peak Pressure": lambda: random.randint(10, 100),
     "FIO2": lambda: random.randint(10, 100),
     "Arterial Line": lambda: random.randint(10, 100),
     "I/E Ration": lambda: random.randint(10, 100),
     "Ispiratory Time": lambda: random.randint(10, 100),
}

ivac = {
     "שם החומר במזרק": lambda: random.choice(['MEDICNE', "WINE", "COFFE", "BEER"]),
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
    qrsize = random.randint(100,150)
    img = cv2.resize(np.array(img,dtype=np.uint8)*255,(qrsize,qrsize))
    img= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    return img, text


def create_segments(device_type, image_size=[1200, 1000], x_start=200, y_start=200):
    x = x_start
    y = y_start
    y_step = 100
    x_step = 400
    segments = []
    for m in devices[device_type]:
        segments.append({
            "top": y,
            "left": x,
            "bottom": y+y_step,
            "right": x+x_step,
            "name": m,
            
        })
        if x+x_step < image_size[1]:
            x += x_step
        else:
            x = x_start
            y += y_step
    return segments


def fill_segments(segments, device_type):
    device = devices[device_type]
    values = []
    colors = []
    for s in segments:
        values.append({"segment_name": s['name'], "value": device[s['name']]()})
        colors.append((random.randint(100,200),random.randint(100,200),random.randint(100,200)))
    return values, colors


def change_values(values):
    for i in range(len(values)):
        if str(values[i]['value']).isnumeric():
            values[i]['value'] += random.randint(-1, 1)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def generate_picture(qrcode, image_size, segments, values, colors, fontScale,thickness):
    image = np.zeros((image_size[0],image_size[1],3),dtype=np.uint8)
    image[12:qrcode.shape[0]+12, 18:qrcode.shape[1]+18, :] = qrcode
    for s, v, color in zip(segments, values,colors):
        image = cv2.putText(img=image, text=str(v['value']), org=(s['left'], s['top']), fontFace=cv2.FONT_HERSHEY_PLAIN, 
            fontScale=fontScale, color=color, thickness=thickness)
    return image

class Device():

    def __init__(self, device_type, patient, room_number):
        self.index = 0
        self.monitor_id = None
        self.device_type = device_type
        self.image_size = [1200, 1000]
        self.segments = create_segments(device_type, self.image_size)
        self.values, self.colors = fill_segments(self.segments, device_type)
        qrcode, self.qrtext = get_qr_code(device_type)
        self.qrcode =  rotate_image(qrcode, float(random.randint(-10, 10)))
        self.patient = patient
        self.room_number = room_number
        
        self.fontScale = random.randint(2, 5)
        self.thickness= random.randint(2, 3)


    def picture(self):
        return generate_picture(
            self.qrcode, self.image_size, self.segments, self.values,self.colors,
            self.fontScale,self.thickness,
            )

    def change_values(self):
        change_values(self.values)


def fill_rooms(device_count):
    active_devices = []
    for i in range(device_count):
        room = random.randint(1, 400)
        patient = random.choice(name_list)
        active_devices.append(Device('ivac', patient, room))
        active_devices.append(Device('monitor', patient, room))
        active_devices.append(Device('respirator', patient, room))
    return active_devices





def send_all_pictures(url, active_devices):
    device_indxes = list(range(len(active_devices)))
    random.shuffle(device_indxes)
    for di in device_indxes:
        device = active_devices[di]
        picutre = device.picture()
        picture = rotate_image(picutre, float(random.randint(-0, 0)))
        b = io.BytesIO()
        imageio.imwrite(b, picture, format='jpeg')
        b.seek(0)
        headers={'Content-Type':'image/jpeg','X-IMAGE-ID':str(device.index)}
        if device.monitor_id is not None:
            headers['X-MONITOR-ID']=str(device.monitor_id)
        res = requests.post(url + '/monitor_image', data=b,headers=headers)
        res_data = res.json()
        print(res_data)
        if 'nextImageId' in res_data:
            device.index = res_data['nextImageId']
        if 'monitorId' in res_data:
            device.monitor_id = 'monitorId'
        time.sleep(0.1)


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
    # for d in active_devices:
    #     cv2.imwrite(d.qrtext+'.jpg', rotate_image(d.picture(), float(random.randint(-0, 0))))
    url = 'http://cvmonitors.westeurope.cloudapp.azure.com'
    url = 'http://52.157.71.156'
    send_all_pictures(url, active_devices)
    add_devices(url, active_devices)
    while True:
        send_all_pictures(url, active_devices)
        for d in active_devices:
            d.change_values()
        time.sleep(1)


if __name__ == "__main__":
    main()
