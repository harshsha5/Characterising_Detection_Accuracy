import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2
import imutils
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
import ipdb


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def get_good_contours(c, new_width, new_height):
    '''
    INPUT: Takes in the contour and the height and width of the image
    OUTPUT: Returns the contours which lie within the specified screen range provided
    '''
    l = []
    for j, elt in enumerate(c):
        rightmost = tuple(elt[elt[:,:, 0].argmax()][0])
        topmost = tuple(elt[elt[:,:, 1].argmin()][0])
        if ((topmost[1] > 3 * new_height / 10 and topmost[1] < 7 * new_height / 10) and (
                rightmost[0] > (0.4 * new_width))):
            l.append(elt)
    return l

def circular_mask(radius, height, width, layer = 3):
    # background = np.ones((height, width))
    # background[int(height/2),int(width/2)] = 23987
    a= int(height/2)
    b = int(width/2)
    y, x = np.ogrid[-a:height-a, -b:width-b]
    mask = x ** 2 + y ** 2 <= radius ** 2
    mask_1 = np.ones((height, width))
    mask_1[mask] = 0
    # mask = -1 * mask.astype(float) + 1
    # mask_1 = convolve(background, mask) - sum(sum(mask)) + 1
    mask_ = np.zeros((height, width,layer))
    for i in range(layer):
        mask_[:,:,i] = mask_1
    return mask_

def get_centroid(img):

    new_height, new_width, channels = img.shape 
    output = img
    PINK_MIN = np.array([75,5,0], np.uint8)
    PINK_MAX = np.array([90,255,255], np.uint8)


    # Apply the pink filter
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    frame_threshed = cv2.inRange(output, PINK_MIN, PINK_MAX)

    frame_threshed = frame_threshed/255.0
    # struct1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    struct1 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    # frame_threshed = scipy.ndimage.morphology.binary_dilation(frame_threshed, iterations=1)
    frame_threshed = scipy.ndimage.morphology.binary_erosion(frame_threshed, structure = struct1, iterations = 1)
    # frame_threshed = scipy.ndimage.morphology.binary_dilation(frame_threshed, iterations = 4)
    frame_threshed = frame_threshed*255.0
    frame_threshed = frame_threshed.astype("uint8")

    '''
    DISPLAYING OF IMAGES AND THRESHOLD
    '''

    # Display the threshed frame and the original image
    # cv2.imwrite("./results2/" + str(i) + ".png", frame_threshed)
    #cv2.imshow('frame',frame_threshed)
    #cv2.imshow('actual_image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    should_display = 1
    if should_display == 1:

        '''
        #DISPLAYING OF IMAGES AND THRESHOLD
        '''

        # Display the threshed frame and the original image
        # cv2.imshow('frame',frame_threshed)
        # cv2.imshow(q'image', img)
        # frame_threshed = cv2.cvtColor(frame_threshed, cv2.COLOR_BGR2GRAY)
        # im2, contours = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2, contours,_ = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            stro = "No contour detected for image"
            print(stro)
            return -1,-1

        # Get good contours
        # contours = get_good_contours(im2, new_width, new_height)
        # if (len(contours) == 0):
        #     stro = "No Good contour detected for image"
        #     print(stro)
        #     return -1,-1


        # Detecting max contour & making sure it is bigger than a threshold
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 0:
            stro = "Detected contour too small"
            print(stro)
            return -1,-1


        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Drawing contours over the original image. Just for validation
        # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

        cX = int(x + (w / 2))
        cY = int(y + (h / 2))

        cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
        cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        '''
        #DISPLAYING OF IMAGES AND THRESHOLD
        '''
        # cv2.imshow("Marking_centroid", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cX,cY

if __name__ == "__main__":
    images_path = sys.argv[1]
    images_path = '../image_data/'  #Remove this

    cX,cY = get_centroid(images_path + filename)



