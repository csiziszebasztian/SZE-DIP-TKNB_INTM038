import cv2
import numpy as np
'''import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import scipy.ndimage as nd
import math
from PIL import Image, ImageDraw
import imutils
import sys
import scipy as sp
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert'''
import pytesseract



font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((3,3), np.uint8)

imgs=[]
img = cv2.imread('test_img/night_stop.jpg', cv2.IMREAD_COLOR)


'''img = cv2.resize(img, (620,480) )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

cnts,hie = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 3:
                      screenCnt = approx
                      continue
                elif len(approx) == 4:
                      screenCnt = approx
                      continue
                elif len(approx) == 8:
                      screenCnt = approx
                      continue

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('new', new_image)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

cv2.imshow('cropp', Cropped)'''

'''canny = cv2.Canny(img, 50, 240)
blur_canny = cv2.blur(canny, (2, 2))
_, thresh_canny = cv2.threshold(blur_canny, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('thresh_canny', thresh_canny)

cnts, hierarchy = cv2.findContours(thresh_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    accuracy = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, accuracy, True)
    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

cv2.imshow('Approx polyDP', img)'''

'''laplacian = cv2.Laplacian(blur, cv2.CV_64F)

laplacian1 = laplacian/laplacian.max() * 10
laplacian1 = np.uint8(laplacian1)

cnts, hierarchy=cv2.findContours(laplacian1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

for cnt in cnts:
    accuracy = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, accuracy, True)
    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

cv2.imshow('Approx polyDP', img)'''


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Red mask
mask_r1 = cv2.inRange(hsv, (0, 127, 100), (12, 255, 255))
mask_r2 = cv2.inRange(hsv, (164, 132, 0), (210, 255, 255))
mask_r = cv2.bitwise_or(mask_r1, mask_r2)
#Blue mask
mask_b = cv2.inRange(hsv, (100, 100, 100), (140, 255, 255))

mask = mask_r + mask_b
#Color segmentation
hsv_out = cv2.bitwise_and(hsv, hsv, mask = mask)
blur_hsv_out = cv2.blur(hsv_out,(1,1))
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


for cnt in cnts:
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(img_dilation, start, end, [255, 255, 255], 1)


cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

for cnt in cnts:

    (x, y), (w, h), angle = cv2.minAreaRect(cnt)

    koeff_p = 0
    if w >= h and h != 0:
        koeff_p = w / h
    elif w != 0:
        koeff_p = h / w
    if koeff_p > 2:
        continue

    M = cv2.moments(cnt)
    cX = 0
    cY = 0
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    x, y, w_b_rect, h_b_rect = cv2.boundingRect(cnt)

    accuracy = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, accuracy, True)
    cv2.drawContours(img, [approx], 0, (0, 255, 0), 1)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), font, 1, (0))
    elif len(approx) == 4:
        cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
    elif len(approx) == 8:
        cv2.putText(img, "Octagon", (x, y), font, 1, (0))
    else:
        cv2.putText(img, "Circle", (x, y), font, 1, (0))

    cv2.rectangle(img, (cX - int(w_b_rect / 2) - 10, cY - int(h_b_rect / 2) - 10),
                      (cX + int(w_b_rect / 2) + 10, cY + int(h_b_rect / 2) + 10), (255, 0, 0), 1)

    crop=img[cY - int(h_b_rect / 2) - 3:cY + int(h_b_rect / 2) + 3, cX - int(w_b_rect / 2) - 3:cX + int(w_b_rect / 2) + 3]


'''text = pytesseract.image_to_string(crop, config='--psm 11')
print("Detected Number is:",text)'''

cv2.imshow('IMG',img)

cv2.waitKey(0)
cv2.destroyAllWindows()