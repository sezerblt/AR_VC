import cv2
import numpy as np
import time
import os
import PaintHandTracking as htm
import unittest


#######################
#settting of brush and eraser 
brushThickness = 25
eraserThickness = 100
########################

#get all imafes file from path
#C:\\Users\\rootx\\Desktop\\Test_Module
f_path = "C:\\Users\\rootx\\Desktop\\Test_Module\\images"
myList = os.listdir(f_path)
print("Loading...")

#set an list of overlay
overlayList = []

#get an aimage from path list and add overrlay list
for imPath in myList:
    image = cv2.imread(f_path+"\\"+imPath)
    overlayList.append(image)

#Test:what is size of overlay and set overlay first element to header
print(len(overlayList))
header = overlayList[0]
#default color
drawColor = (255, 0, 255)
#get videocapture class and adding inner camera(int 0)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#call create paintdetector object,create start points,create canvas matrix(3channels:RGB)
detector = htm.PaintHandDetector(detectionCon=0.65,maxHands=2)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

#start loop
while True:
    # 1. read capture object
    success, img = cap.read()
    #flipping camera optics
    img = cv2.flip(img, 1)
    #detecting landmark
    #detector object return img with finds shapes
    img = detector.findHands(img)
    #finding marks of hand and assigmnet of a landMarks value
    lmList = detector.findPosition(img, draw=False)
    #is not equal  0(available-> lmlist)
    if len(lmList) != 0:
        # set value points:second finger all points,middle finger all points,
        
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3.is upping fingers
        fingers = detector.fingersUp()
        # print(fingers)

        #is finger 2 and 3 upped
        if fingers[1] and fingers[2]:
            #init dot area1draw RECTANGLE
            xp, yp = 0, 0
            #print("Selection Mode")
            #y1 lessthan 125
            if y1 < 125:
                if 250 < x1 < 450:
                    #select first overlay and color is red
                    header = overlayList[0]
                    drawColor = (0, 0, 250)
                elif 550 < x1 < 750:
                    #select secınd overlay and color is blue
                    header = overlayList[1]
                    drawColor = (250, 0, 0)
                elif 800 < x1 < 950:
                    #select third overlay and color is green
                    header = overlayList[2]
                    drawColor = (0, 250, 0)
                elif 1050 < x1 < 1200:
                    #select third overlay and color is black and eraser mode
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
        
            cv2.rectangle(img, (x1, y1-25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. cizim modu
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
        
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            #cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            #------------------------------------

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            #-------------------
            xp, yp = x1, y1

# # Clear Canvas when all fingers are up
# if all (x >= 1 for x in fingers):
# imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    # Setting the header image and
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("MyTest-AR version 1,0", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
