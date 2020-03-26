#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
from uuid import uuid4
import qrcode
import random
import numpy as np
import requests
import time
import pickle
import os
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
     "שם החומר במזרק": lambda: random.choice(['תרופה א', "תרופה כלשהי", "ערק", "קפה"]),
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
    r = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
    text = f'cvmonitors-{title}-{uuid4().hex[:16]}'
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    return img, text


def create_segments(device_type, image_size=[1200, 1000], x_start=200, y_start=200):
    x = x_start
    y = y_start
    y_step = 100
    x_step = 100
    segments = []
    for m in devices[device_type]:
        segments.append({
            "top": y,
            "left": x,
            "bottom": y+y_step,
            "right": x+x_step,
            "name": m
        })
        if x+x_step < image_size[1]:
            x += x_step
        else:
            x = x_start
            y += y_start
    return segments
    class Device():


def fill_segments(segments, device_type):
    device = devices[device_type]
    values = []
    for s in segments:
        values.append({"segment_name": s, "value": device[s]()})


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


def generate_picture(qrcode, image_size, segments, values):
    image = np.zeros(image_size)
    image[12:qrcode.shape[0]+12, 18:qrcode.shape[1]+18, :] = qrcode
    for s, v in zip(segments, values):
        cv2.putText(image, s['v'], (s['left'], s['top'], cv2.FONT_HERSHEY_COMPLEX, fontScale=random.randint(20, 40)))


class Device():

    def __init__(self, device_type, patient, room_number):
        self.device_type = device_type
        self.image_size = [1200, 1000]
        self.segments = create_segments(device_type, self.image_size)
        self.values = fill_segments(segments, device_type)
        self.qrcode, self.qrtext = rotate_image(
            get_qr_code(device_type), float(random.randint(-20, 20)))
        self.patient = patient
        self.room_number = room_number

    def picture(self):
        return generate_picture(self.qrcode, self.image_size, self.segments, self.values)

    def change_values():
        change_values(values)


def fill_rooms(device_count):
    active_devices = []
    for i in range(device_count):
        room = random.randint(1, 400)
        patient = random.choice(name_list)
        active_devices.append(Device('ivac', patient, room))
        active_devices.append(Device('monitor', patient, room))
        active_devices.append(Device('respirator', patient, room))
    return active_devices


def send_all_pictures(active_devices):
    device_indxes = list(range(len(active_devices)))
    random.shuffle(device_indxes)
    for di in device_indxes:
        picutre = active_devices[di].picture
        picture = rotate_image(picutre, float(random.randint(-15, 15)))
        jpg = cv2.imencode('.jpg', picutre)
        res = requests.post(url, data=jpg, headers={
                            'content-type': 'image/jpeg'})
        time.sleep(100)


def add_devices(url, active_devices):

    for di in range(len(active_devices)):
        device = active_devices[di]
        device_json = {
            "monitorId": device.qrtext,
            "patientId": device.patient,
            "roomId": device.room_number
            "deviceCategory": device.device_type
            "segments": device.segments
        }
        requests.post(url + f'/monitor/{device.qrtext}', json=device_json)


def main()
   if os.path.exists('devices.pkl'):
        active_devices = pickle.load(open('devices.pkl', 'wb'))
    else:
        active_devices = fill_rooms()
        pickle.dump(active_devices, open('devices.pkl','wb'))
    return
    url = 'http://cvmonitors.westeurope.cloudapp.azure.com'
    send_all_pictures()
    add_devices(active_devices)
    return
    while True:
        send_all_pictures()
        for d in active_devices:
            if random.randint(0,10) <3:
                d.change_values()
        time.sleep(1)


if if __name__ == "__main__":
    main()
