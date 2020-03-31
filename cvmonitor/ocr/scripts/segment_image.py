
import cv2
from mouse_crop import mouse_crop

if __name__ == '__main__':

    img_path = r'/home/moshes2/datasets/monitors/BneiZion2/1.tiff'

    img = cv2.imread(img_path, -1)

    roi, coordinates = mouse_crop(img, num_of_crops=3)

    print('Done!')
