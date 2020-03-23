# %%
from pyzbar import pyzbar
import pylab
import imageio
import numpy as np
import cv2
# %%
image = open(os.path.dirname(__file__)+'/data/barcode.png', 'rb').read()
image = np.asarray(imageio.imread(image))
decodedObjects = pyzbar.decode(image)
pylab.imshow(image)

# %%
qrDecoder = cv2.QRCodeDetector()
image = open(os.path.dirname(__file__)+'/data/qrcode.png','rb').read()
image = np.asarray(imageio.imread(image))
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(image)
pylab.imshow(rectifiedImage)


# %%
