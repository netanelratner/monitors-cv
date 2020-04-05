import random
import numpy as np


def get_ocr_expected_boxes(segments, devices, default_score, min_score_to_reprocess):
    """
    Create expected boxes from segments.
    Returns the boxes to perform ocr on them. each box will have the data
    needed to run ocr, and will contain the original segment index
    """
    expected_boxes = []
    for index, segment in enumerate(segments):
        expected = {
            "bbox": [
                segment["left"],
                segment["top"],
                segment["right"],
                segment["bottom"],
            ],
            "name": segment["name"],
            "index": index,
        }
        needs_ocr = True
        if "value" in segment and "name" in segment:
            value = segment["value"]
            name = segment["name"]
            device_params = devices.get(name)
            score = segment.get("score", default_score)
            if (
                device_params is not None
                and is_text_valid(value, device_params)
                and score > min_score_to_reprocess
            ):
                needs_ocr = False
        if needs_ocr:
            expected_boxes.append(expected)
    return expected_boxes



def is_text_valid(text, device_name_params):

    """
    Simple text validation.
    Currently validates only numeric values.
    Verified characters:
        - number of digits
        - minimal and maximal values
        
    Parameters
        ----------
        text : str
            Text to be verified
        device_name_params : dict
            Dictionary defines device parameters (as written in get_fields_info())

        Returns
        -------
        is_valid : bool
            True if text is valid, False otherwise
    """

    is_valid = True

    dtype = device_name_params.get('dtype',str)

    if dtype == int or dtype == float:

        if not is_number(text):  # text must be int or float
            is_valid = False
            return is_valid

        val = dtype(text)

        if (device_name_params['min'] is not None) and (val < device_name_params['min']) \
                or \
                (device_name_params['max'] is not None) and (val > device_name_params['max']):
            is_valid = False
            return is_valid

        return is_valid


def get_field_rand_value(field_info, current=None):
    if field_info['dtype'] in [float, int]:
        max_val = field_info.get('max') or int(0.99999 *(10**(field_info['max_len'])))
        min_val = field_info.get('min') or 0
        if current is None:
            base = random.randint(min_val,  max_val)
            if 'num_digits_after_point' in field_info:
                base = float(base) / 10**field_info['num_digits_after_point']
        else:
            base = field_info['dtype'](current) + random.randint(-3,3)
            if random.randint(0,1) == 0:
                base += 0.1
            base = max(min_val,min(base, max_val))
            base =  field_info['dtype'](base)
    if field_info['dtype'] in [str]:
        base = random.choice(['wine','beer','coffee','soda','water'])
    max_len = field_info.get('max_len',20)
    base = str(base)
    if '.' in base:
        max_len+=1
    if len(base) > max_len:
        base = base[:max_len]
    return base

def get_fields_info(device_types=['respirator','ivac','monitor']):

    ivac = {
    'Medication Name': {'max_len': 10, 'dtype': str},
    'Volume Left to Infuse':  {'max_len': 3, 'min': 10, 'max': None, 'dtype': int},
    'Volume to Insert':  {'max_len': 3, 'min': 10, 'max': None, 'dtype': int},
    'Infusion Rate':  {'max_len': 4, 'min': 0, 'max': None, 'dtype': float, 'num_digits_after_point': 1},
    }
    respirator = {
    'Ventilation Mode': {'max_len': 10, 'dtype': str},
    'Tidal Volume': {'max_len': 3, 'min': 350, 'max': 600, 'dtype': int},
    'Expiratory Tidal Volume': {'max_len': 3, 'min': None, 'max': None, 'dtype': int},
    'Rate': {'max_len': 2, 'min': 10, 'max': 40, 'dtype': int},
    'Total Rate': {'max_len': 2, 'min': 10, 'max': 40, 'dtype': int},
    'Peep': {'max_len': 2, 'min': None, 'max': None, 'dtype': int},
    'Ppeak': {'max_len': 2, 'min': None, 'max': 40, 'dtype': int},
    'FIO2': {'max_len': 3, 'min': None, 'max': None, 'dtype': int},
    'I:E Ratio': {'max_len': 2, 'min': None, 'max': None, 'dtype': float, 'num_digits_after_point': 1}, # FIXME: assume that operator selects only X.X part, without the digit 1
    'Inspiratory time': {'max_len': 2, 'min': None, 'max': None, 'dtype': float, 'num_digits_after_point': 1},
    }
    monitor = {
    'HR': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': int},
    'SpO2': {'max_len': 3, 'min': 90, 'max': None, 'dtype': int},
    'RR': {'max_len': 2, 'min': 8, 'max': 26, 'dtype': int},
    'IBP-Systole': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': int},  # left blood pressure
    'IBP-Diastole': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': int},  # right blood pressure
    'NIBP-Systole': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': int},  # left blood pressure
    'NIBP-Diastole': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': int},  # right blood pressure
    'Temp': {'max_len': 3, 'min': 35.0, 'max': 38.0, 'dtype': float, 'num_digits_after_point': 1},
    'etCO2': {'max_len': 2, 'min': 24, 'max': 44, 'dtype': int},
    }
    # for annotations only, currently not found in android app
    #'hr_saturation': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': int},
    devices = {}

    if 'respirator' in  device_types:
        devices.update(respirator)
    if 'monitor' in  device_types:
        devices.update(monitor)
    if 'ivac' in  device_types:
        devices.update(ivac)
    return devices


def annotation_names_mapping():

    """
    anns: field names as given in (matlab) annotations
    names: field names as given in android setting app
    """

    names2anns = {}

    names2anns['hr'] = 'HR'
    names2anns['hr_saturation'] = 'hr_saturation'
    names2anns['spo2'] = 'SpO2'
    names2anns['rr'] = 'RR'
    names2anns['bp_1'] = 'IBP-Systole'
    names2anns['bp_2'] = 'IBP-Diastole'
    names2anns['ibp_1'] = 'IBP-Systole'
    names2anns['ibp_2'] = 'IBP-Diastole'
    names2anns['nibp_1'] = 'NIBP-Systole'
    names2anns['nibp_2'] = 'NIBP-Diastole'
    names2anns['temp'] = 'Temp'
    names2anns['etco2'] = 'etCO2'
    names2anns['screen'] = None

    anns2names = {}
    for key, val in names2anns.items():
        if val is not None:
            anns2names[val] = key

    return names2anns, anns2names

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

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

    names2anns, anns2names = annotation_names_mapping()

    expected_boxes = []

    for key, val in ann_dict.items():

        name = names2anns[key]

        bbox = change_box_type(val['box'], type_in='ltwh', type_out='ltrb')

        bbox = enlarge_box(bbox, percent=0.2)

        expected_box = {'name': name,
                        'bbox': bbox,
                        'true_val': val['val']}

        expected_boxes.append(expected_box)

    return expected_boxes
