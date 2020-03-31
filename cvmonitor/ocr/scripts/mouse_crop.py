import cv2
import numpy as np

def mouse_crop(image, num_of_crops, text='select next ROI'):

    global x_start, y_start, x_end, y_end, cropping, rois, Finishrois, coordinates

    cropping = False
    Finishrois = 0
    rois = []
    coordinates = []
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    def mouse_crop_aux(event, x, y, flags, param):

        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping, rois, Finishrois, coordinates

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:

            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found

                if x_start > x_end:
                    tmp = x_start
                    x_start = x_end
                    x_end = tmp

                if y_start > y_end:
                    tmp = y_start
                    x_start = y_end
                    y_end = tmp

                rois.append(image[y_start:y_end, x_start:x_end])
                coordinates.append([x_start, y_start, x_end, y_end])
                #cv2.imshow("Cropped", rois[-1])
                Finishrois += 1

                # display cropped ROI in different window
                cv2.imshow("Cropped", rois[-1])

                # display bbox on input image
                cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    # declare image
    cv2.namedWindow("image")

    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0, 0, 0)
    cv2.putText(image, text, (10, 50), font, 1, color, 1)

    cv2.imshow("image", image)
    
    while Finishrois < num_of_crops:
        
        cv2.setMouseCallback("image", mouse_crop_aux)

        # show bbox during crop operation
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            i = image.copy()
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        key = cv2.waitKey(1)

    # close all open windows
    cv2.destroyAllWindows()


    return rois, coordinates


if __name__ == '__main__':


    img_path = r'/home/moshes2/datasets/monitors/BneiZion2/1.tiff'

    img = cv2.imread(img_path, -1)

    roi, coordinates = mouse_crop(img, num_of_crops=3)

    print('Done!')