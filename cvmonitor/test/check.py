# %%
from pyzbar import pyzbar
import pylab
import imageio
import numpy as np
import cv2
# %%
image = open(os.path.dirname(__file__)+'/data/qrcode_rotated.png', 'rb').read()
image = np.asarray(imageio.imread(image))
decodedObjects = pyzbar.decode(image)
pylab.imshow(image)


# %%
decodedObjects

# %%
