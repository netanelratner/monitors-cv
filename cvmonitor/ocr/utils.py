


def get_device_fields():

    device_fields = {}

    device_fields = {
                     # monitor
                     'hr': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': 'int'},
                     'hr_saturation': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': 'int'},
                     'spo2': {'max_len': 3, 'min': 90, 'max': None, 'dtype': 'int'},
                     'rr': {'max_len': 2, 'min': 8, 'max': 26, 'dtype': 'int'},
                     'bp_1': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': 'int'},  # left blood pressure
                     'bp_2': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': 'int'},  # right blood pressure
                     # 'ibp_1': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': 'int'},  # left blood pressure
                     # 'ibp_2': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': 'int'},  # right blood pressure
                     # 'nibp_1': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': 'int'},  # left blood pressure
                     # 'nibp_2': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': 'int'},  # right blood pressure
                     'temp': {'max_len': 3, 'min': 35.0, 'max': 38.0, 'dtype': 'float', 'num_digits_after_point': 1},
                     'etco2': {'max_len': 2, 'min': 24, 'max': 44, 'dtype': 'int'},

                     # respirator
                     # TODO

                     # ivac
                     # TODO
                     }

    return device_fields