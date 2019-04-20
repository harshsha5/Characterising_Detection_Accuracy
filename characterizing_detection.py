import numpy as np
import cv2
import ipdb 
import os
import glob
import csv
import matplotlib.pyplot as plt
import sys
import detection_v2 as dt
import imutils


'''
INPUT: Image of the rover with marked centroid (by algo). 
       Range at which that image was captured (set a default value incase range is not received)
DISPLAY: Intermediate results for the video (to be shown in demonstration)
Output: Percentage of correct detection

Receive image inputs with centroid coorfinates determined by the algorithm. The program displays the image on the screen.
Use the point and click code to identify the true centroid of the rover.
Set a threshold of euclidian distance between these two rovers (as a function of range). If the euclidian distanc eis above a certain threshold mark it as 
incorrectly detected else as correctly detected. 
'''

def point_click(img):

    _, ax = plt.subplots(figsize=(9, 9))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = ax.imshow(img)
    x_c, y_c = plt.ginput(1, mouse_stop=2)[0]       #y_c is the row while x_c is the column
    print(y_c,x_c)
    cv2.destroyAllWindows()

    return y_c,x_c

def find_detection_rate(error_array,distance_metric):
    print(error_array)
    correct_detections = len(error_array[np.where( error_array < distance_metric)])
    return correct_detections/error_array.shape[0]

if __name__ == "__main__":

    images_path = sys.argv[1]
    images_path = '../image_data/'  #Remove this and instead take this as a command line argument
    distance_metric = 80 #(in pixels)
    flag = 0
    items = os.listdir(images_path)
    if('.DS_Store' in items):
        items.remove('.DS_Store')


    for i,filename in enumerate(items):

        img = cv2.imread(images_path + filename,-1)
        new_height = int(600)
        new_width = int(900)

        img = imutils.resize(img, height=new_height, width=new_width)

        cX,cY = dt.get_centroid(img)
        y_pc,x_pc = point_click(img)
        print("Centroid detected by algo is: ",cY,"\t",cX,"\n")
        print("Centroid detected by point and click is: ",y_pc,"\t",x_pc,"\n")
        print("\n ===================================================================================================== \n")

        if(flag==0 and cX!=-1):
            flag+=1
            centroid_using_algorithm = np.array([cY,cX])
            centroid_using_point_click = np.array([y_pc,x_pc])
        elif(cX!=-1):
            centroid_using_algorithm = np.vstack((centroid_using_algorithm,np.array([cY,cX])))
            centroid_using_point_click = np.vstack((centroid_using_point_click,np.array([y_pc,x_pc])))

    assert  centroid_using_algorithm.shape == centroid_using_point_click.shape, "Shapes of the two arrays don't match"
    error_array = np.linalg.norm(np.abs(centroid_using_point_click - centroid_using_algorithm),axis = 1)
    detection_rate = find_detection_rate(error_array,distance_metric)
    print("Detection rate is ",detection_rate*100," % \n")






