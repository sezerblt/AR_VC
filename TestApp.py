from FaceTracking import FaceDetector
from HandTrackingClass import HandDetector
import cv2
import time

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    #cap.set(3,880)
    #cap.set(4,880)
    
    detector = HandDetector()
    #-----
    detector2 = FaceDetector()
    #-------
    while True:
        success, img = cap.read()
        #img=cv2.resize(img,(880,660))
        img=cv2.flip(img,1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        #-----
        img, bboxs = detector2.findFaces(img)
        #-------
        #if len(lmList) != 0:
        #    print(lmList[1])
        #   pass

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (200, 50, 250), 3)
        

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
