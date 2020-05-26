import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import cv2
import argparse
from keras.models import load_model

model = load_model('traffic_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing veh > 3.5 tons' }


def Detect_Traffic_Shape(cnt):

    accuracy = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, accuracy, True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    print(len(approx))
    if len(approx) == 3 or len(approx) == 4 or len(approx) == 8:
        return True
    else:
        return False


def ROI_Center(cnt):

    M = cv2.moments(cnt)
    cX = 0
    cY = 0
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    x, y, w_b_rect, h_b_rect = cv2.boundingRect(cnt)

    return cX, cY, x , y, w_b_rect , h_b_rect


def Crop(cnt,output):

    cX, cY, x , y, w_b_rect , h_b_rect=ROI_Center(cnt)

    crop = output[cY - int(h_b_rect / 2) - 3:cY + int(h_b_rect / 2) + 3,
           cX - int(w_b_rect / 2) - 3:cX + int(w_b_rect / 2) + 3]

    return crop


def DrawBox(cnt,output,sign):

    cX, cY, x , y, w_b_rect , h_b_rect=ROI_Center(cnt)

    cv2.rectangle(output, (cX - int(w_b_rect / 2) - 10, cY - int(h_b_rect / 2) - 10),
                  (cX + int(w_b_rect / 2) + 10, cY + int(h_b_rect / 2) + 10), (255, 0, 0), 1)

    cv2.putText(output, sign, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)


def classify(img):
    global label_packed
    img = cv2.resize(img,(30,30), interpolation = cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    pred = model.predict_classes([img])[0]
    sign = classes[pred+1]
    print(sign)
    return sign


def Detect_Circle(img_thresh,img_out, original):



    circles = cv2.HoughCircles(img_thresh, cv2.HOUGH_GRADIENT, 1.2, 500)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            cv2.rectangle(img_out, (x - r, y - r), (x + r, y + r), (0, 128, 255), 1)
            cv2.circle(img_out, (x, y), r, (0, 255, 0), 2)
            crop=original[y-r:y+r,x-r:x+r]
            sign=classify(crop)
            cv2.putText(img_out, sign, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

    return img_out





def detection(img_name):

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask
    mask_rl = cv2.inRange(hsv, (0, 127, 100), (12, 255, 255))
    mask_rh = cv2.inRange(hsv, (164, 132, 0), (210, 255, 255))
    mask_r = cv2.bitwise_or(mask_rl, mask_rh)
    # Blue mask
    mask_b = cv2.inRange(hsv, (100, 100, 100), (140, 255, 255))
    #Yellow mask
    mask_y = cv2.inRange(hsv,(15,110,110),(25,255,255))

    mask = mask_r + mask_y + mask_b

    # Color segmentation
    hsv_out = cv2.bitwise_and(hsv, hsv, mask=mask)
    blur_hsv_out = cv2.blur(hsv_out, (1, 1))
    gray = cv2.cvtColor(blur_hsv_out, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)



    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if (area < 500):
            cv2.fillPoly(thresh, pts=[cnt], color=(0, 0, 0))
            continue
        cv2.fillPoly(thresh, pts=[cnt], color=(255, 255, 255))

    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    cnts, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    '''for cnt in cnts:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(img_dilation, start, end, [255, 255, 255], 1)

    cnts, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]'''

    cv2.imshow('t', img_dilation)

    output = img.copy()

    output=Detect_Circle(img_dilation, output, img)


    for cnt in cnts:

        (x, y), (w, h), angle = cv2.minAreaRect(cnt)

        koeff_p = 0
        if w >= h and h != 0:
            koeff_p = w / h
        elif w != 0:
            koeff_p = h / w
        if koeff_p > 2:
            continue


        if Detect_Traffic_Shape(cnt):
            crop=Crop(cnt,img)
            sign=classify(crop)
            DrawBox(cnt,output,sign)

    return output






ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
img=cv2.imread(args["image"])

##output=detection('inputs/foutvonal.jpg')
cv2.imshow('output', output)

cv2.waitKey(0)
cv2.destroyAllWindows()


