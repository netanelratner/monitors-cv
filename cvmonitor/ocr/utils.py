


def get_device_names():

    device_names = {}

    device_names =  {
                     # ivac
                     'Medication Name': {'max_len': 10, 'dtype': 'str'},
                     'Volume Left to Infuse':  {'max_len': 3, 'min': 10, 'max': None, 'dtype': 'int'},
                     'Volume to Insert':  {'max_len': 3, 'min': 10, 'max': None, 'dtype': 'int'},
                     'Infusion Rate':  {'max_len': 4, 'min': 0, 'max': None, 'dtype': 'float', 'num_digits_after_point': 1},

                     # respirator
                     'Ventilation Mode': {'max_len': 10, 'dtype': 'str'},
                     'Tidal Volume': {'max_len': 3, 'min': 350, 'max': 600, 'dtype': 'int'},
                     'Expiratory Tidal Volume': {'max_len': 3, 'min': None, 'max': None, 'dtype': 'int'},
                     'Rate': {'max_len': 2, 'min': 10, 'max': 40, 'dtype': 'int'},
                     'Total Rate': {'max_len': 2, 'min': 10, 'max': 40, 'dtype': 'int'},
                     'Peep': {'max_len': 2, 'min': None, 'max': None, 'dtype': 'int'},
                     'Ppeak': {'max_len': 2, 'min': None, 'max': 40, 'dtype': 'int'},
                     'FIO2': {'max_len': 3, 'min': None, 'max': None, 'dtype': 'int'},
                     'I:E Ratio': {'max_len': 2, 'min': None, 'max': None, 'dtype': 'float', 'num_digits_after_point': 1}, # FIXME: assume that operator selects only X.X part, without the digit 1
                     'Inspiratory time': {'max_len': 2, 'min': None, 'max': None, 'dtype': 'float', 'num_digits_after_point': 1},

                     # monitor
                     'Heart Rate': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': 'int'},
                     'SpO2': {'max_len': 3, 'min': 90, 'max': None, 'dtype': 'int'},
                     'RR': {'max_len': 2, 'min': 8, 'max': 26, 'dtype': 'int'},
                     'IBP-Systole': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': 'int'},  # left blood pressure
                     'IBP-Diastole': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': 'int'},  # right blood pressure
                     'NIBP-Systole': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': 'int'},  # left blood pressure
                     'NIBP-Diastole': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': 'int'},  # right blood pressure
                     'Temp': {'max_len': 3, 'min': 35.0, 'max': 38.0, 'dtype': 'float', 'num_digits_after_point': 1},
                     'etCO2': {'max_len': 2, 'min': 24, 'max': 44, 'dtype': 'int'},

                     # for annotations only, currently not found in android app
                     'hr_saturation': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': 'int'},

                     }

    return device_names


def annotation_names_mapping():

    """
    anns: field names as given in (matlab) annotations
    names: field names as given in android setting app
    """

    names2anns = {}

    names2anns['hr'] = 'Heart Rate'
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
