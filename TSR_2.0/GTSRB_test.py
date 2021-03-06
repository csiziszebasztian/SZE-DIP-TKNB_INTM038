import numpy as np
import cv2
import argparse
import imutils
from keras.models import load_model
from PIL import ImageTk, Image
import os

model = load_model('traffic_classifier.h5')

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
           43:'End no passing veh > 3.5 tons'}


def detection_test():
    c_range = 10
    c=[0,5,11,12,13,14,15,18,33,34]
    cur_path = os.getcwd()
    count=0

    full_img = 0
    full_good = 0
    full_bad = 0
    full_accuracy = 0.0

    for i in range(c_range):
        path = os.path.join(cur_path, 'Train', str(c[i]))
        images = os.listdir(path)
        class_img = 0
        good = 0
        bad = 0
        pred = 0
        class_accuracy = 0.0
        for a in images:
            try:
                img = Image.open(path + '\\' + a)
                w,h=img.size
                if w >= 70 and h >= 70:
                 detect=detection(img,a)
                 class_img = class_img + 1
                 if detect == 1:
                     good += 1
                 else:
                     bad += 1
            except:
                print("Error loading image")

        class_name=c[count] + 1
        count+=1
        class_accuracy = (good * 100) / class_img
        class_accuracy = "{:.2f}".format(class_accuracy)
        print(classes[class_name])
        print("######################")
        print(class_img)
        print(good)
        print(bad)
        print(class_accuracy)
        print("######################")
        full_img = full_img + class_img
        full_good+=good
        full_bad+=bad


    full_accuracy= (full_good * 100) / full_img
    full_accuracy = "{:.2f}".format(full_accuracy)
    print(full_img)
    print(full_good)
    print(full_bad)
    print (full_accuracy)

def classification_test():
    c = 43
    cur_path = os.getcwd()

    full_img=0
    full_good=0
    full_bad=0
    full_accuracy=0.0

    for i in range(c):
        path = os.path.join(cur_path, 'Train', str(i))
        images = os.listdir(path)
        class_img = 0
        good = 0
        bad = 0
        pred=0
        class_accuracy=0.0
        for a in images:
            try:
                img = Image.open(path + '\\'+ a)
                pred=classify(img)
                class_img=class_img+1
                if pred==i:
                    good+=1
                else:
                     bad+=1
            except:
                 print("Error loading image")

        class_accuracy = (good * 100) / class_img
        class_accuracy = "{:.2f}".format(class_accuracy)
        print(classes[pred+1])
        print("######################")
        print(class_img)
        print(good)
        print(bad)
        print(class_accuracy)
        print("######################")
        full_img = full_img + class_img
        full_good+=good
        full_bad+=bad


    full_accuracy= (full_good * 100) / full_img
    full_accuracy = "{:.2f}".format(full_accuracy)
    print(full_img)
    print(full_good)
    print(full_bad)
    print (full_accuracy)




def Detect_Traffic_Shape(cnt,color):


    min_bound_rect = 0
    bound_rect = 0
    area = 0
    rec = 0
    komp = 0
    perimeter = 0
    cir = 0
    asect_ratio1=0
    asect_ratio2=0


    area = cv2.contourArea(cnt)

    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    min_bound_rect = w * h
    if h==0 or w==0:
        return False
    asect_ratio1 = w / h

    x, y, w, h = cv2.boundingRect(cnt)
    bound_rect = w * h
    if h==0 or w==0:
        return False
    asect_ratio2 = w / h

    perimeter = cv2.arcLength(cnt, True)

    if min_bound_rect > 0:
        rec = area / min_bound_rect

    if area > 0:
        komp = (perimeter * perimeter) / area
        cir = area / (perimeter * perimeter)


    if color=='yellow' or color=='blue':
        if (komp >= 13.0 and komp <= 25.0) and ((rec >= 0.85 and rec <= 1.1) or (cir >= 0.06 and cir <= 0.07)) and ((asect_ratio1 >= 0.85 and asect_ratio1 <=1.1) or (asect_ratio2 >= 0.85 and asect_ratio2 <=1.1)):
            return True
    elif color=='red':
        if (komp >= 13.0 and komp <= 25.0) and ((rec >= 0.85 and rec <= 1.1) or (cir >= 0.06 and cir <= 0.07)) and ((asect_ratio1 >= 0.85 and asect_ratio1 <= 1.1) or (asect_ratio2 >= 0.85 and asect_ratio2 <= 1.1)):
            return True
        elif (komp >= 13.0 and komp <= 25.0) and ((rec <= 0.50) or (cir < 0.06)) and ((asect_ratio1 >= 0.85 and asect_ratio1 <= 1.1) or (asect_ratio2 >= 0.85 and asect_ratio2 <= 1.1)):
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
    ##img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ##img = Image.fromarray(img)
    img = img.resize((30,30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    pred = model.predict_classes([img])[0]
    #sign = classes[pred+1]
    return pred



def threshold(mask,hsv):

    kernel = np.ones((1, 1), np.uint8)

    hsv_out = cv2.bitwise_and(hsv, hsv, mask=mask)
    blur_hsv_out = cv2.blur(hsv_out, (1, 1))
    gray = cv2.cvtColor(blur_hsv_out, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)


    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


    items = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(items)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    cv2.drawContours(closing, cnts, contourIdx=-1, color=(255, 255, 255), thickness=-1)

    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    erosion = cv2.erode(opening, kernel, iterations=2)

    return  erosion

def detection(img, img_name):

    numpy_image = np.array(img)
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask
    mask_rl = cv2.inRange(hsv, (0, 127, 100), (12, 255, 255))
    mask_rh = cv2.inRange(hsv, (164, 132, 0), (210, 255, 255))
    mask_r = cv2.bitwise_or(mask_rl, mask_rh)
    # Blue mask
    mask_b = cv2.inRange(hsv, (100, 100, 100), (140, 255, 255))
    #Yellow mask
    mask_y = cv2.inRange(hsv,(15,110,110),(25,255,255))


    red_thresh=threshold(mask_r,hsv)
    blue_thresh=threshold(mask_b, hsv)
    yellow_thresh=threshold(mask_y,hsv)

    output = img.copy()

    detect = 0

    cnts, hierarchy = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    cnts, hierarchy = cv2.findContours(blue_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    cnts, hierarchy = cv2.findContours(yellow_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]



    for cnt in red_cnts:
        if Detect_Traffic_Shape(cnt,'red'):
           detect+=1

    for cnt in blue_cnts:
        if Detect_Traffic_Shape(cnt,'blue'):
            detect+=1

    for cnt in  yellow_cnts:
        if Detect_Traffic_Shape(cnt, 'yellow'):
            detect+=1


    return detect


print('Classification Test:\n')
classification_test()
print('Detection Test:\n')
detection_test()



