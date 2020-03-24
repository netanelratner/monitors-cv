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


#%%
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    # dst = np.array([
    #     [0,0],
    #     [image.shape[1] - 1, 0],
    #     [image.shape[1] - 1, image.shape[0] - 1],
    #     [0, image.shape[0] - 1]], dtype = "float32")
        
    # compute the perspective transform matrix and then apply it
    print(f'rect: {rect}')
    print(f'dst: {dst}')
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = np.linalg.inv(M)
    a = Minv @ np.array((0, 0, 1))
    a /= a[-1]

    b = Minv @ np.array((0, image.shape[1], 1))
    b /= b[-1]

    c = Minv @ np.array((image.shape[0],image.shape[1], 1))
    c /= c[-1]

    d = Minv @ np.array([image.shape[0], 0, 1])
    d /= d[-1]

    x = np.ceil(np.abs(min(min(a[0], b[0]), min(c[0], d[0]))))
    y = np.ceil(np.abs(min(min(a[1], b[1]), min(c[1], d[1]))))

    width = int(np.ceil(abs(max(max(a[0], b[0]), max(c[0], d[0])))) + x)
    height = int(np.ceil(abs(max(max(a[1], b[1]), max(c[1], d[1])))) + y)

    rect += np.array([x,y])


    
    warped = cv2.warpPerspective(image, M, (width, height))
    # return the warped image
    return warped

# % %
qrDecoder = cv2.QRCodeDetector()

image = open(os.path.dirname(__file__)+'/data/barcode_monitor.jpg','rb').read()
image = np.asarray(imageio.imread(image))
#image = image[100:400,100:400,:]
#pylab.imshow(image)
#pylab.show()
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
image = clahe.apply(image)
#pylab.imshow(image,cmap='gray')
#pylab.show()
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(image)
decodedObjects = pyzbar.decode(image)
print(f'rect? {rectifiedImage} {decodedObjects}')
print(decodedObjects[0].polygon)
pts= np.array([(p.x,p.y) for p in decodedObjects[0].polygon])
warped = four_point_transform(image, pts)
pylab.imshow(warped,cmap='gray')
pylab.show()

#%%


pylab.imshow(rectifiedImage)


# %%
