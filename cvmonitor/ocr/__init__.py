import os
from urllib import parse
import requests
import hashlib
import logging
from tqdm import tqdm


def download_model(url, filename, hash, basedir="/PreTrained/"):
    path = os.path.dirname(__file__) + basedir + filename

    def download():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            result = requests.get(url, stream=True)
            # Total size in bytes.
            total_size = int(result.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=filename)
            for data in result.iter_content(block_size):
                t.update(len(data))
                f.write(data)
            t.close()

    def is_model_ok():
        m = hashlib.md5()
        m.update(open(path, "rb").read())
        md5sum = m.hexdigest()
        if hash is not None and hash != md5sum:
            logging.error("wrong model checksum.")
            return False
        return True

    if not os.path.exists(path):
        download()

    if is_model_ok():
        logging.info("model found.")
        return path
    else:
        raise RuntimeError("Could not get model")


def get_models():
    models = dict(
        tes_eng=(
            "https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata",
            "tessdata/eng.traineddata",
            None,
        ),
        tes_osd=(
            "https://github.com/tesseract-ocr/tessdata/raw/master/osd.traineddata",
            "tessdata/osd.traineddata",
            None,
        ),
        tps=(
            "https://cvmonitormodelstorage.blob.core.windows.net/cvmodels/TPS-ResNet-BiLSTM-Attn.pth",
            "TPS-ResNet-BiLSTM-Attn.pth",
            "2d0c1fe9e71fa5104a74137971857c88",
        ),
    )
    for name, loc_data in models.items():
        models[name] = download_model(*loc_data)

    return models
