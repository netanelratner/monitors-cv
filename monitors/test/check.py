# %%
from pyzbar import pyzbar
import pylab
import imageio
import numpy as np
# %%
image = open(os.path.dirname(__file__)+'/data/barcode.png', 'rb').read()
image = np.asarray(imageio.imread(image))
decodedObjects = pyzbar.decode(image)
pylab.imshow(image)

# %%
