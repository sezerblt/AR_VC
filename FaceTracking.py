import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    text="%"+str(int(detection.score[0] * 100))+" human"

                    cv2.putText(img,text,
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            1, (250, 0, 0), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (25, 25, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (25, 250, 250), t)
        cv2.line(img, (x, y), (x, y+l), (25, 250, 250), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (25, 250, 250), t)
        cv2.line(img, (x1, y), (x1, y+l), (25, 250, 250), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (25, 250, 250), t)
        cv2.line(img, (x, y1), (x, y1 - l), (25, 250, 250), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (25, 250, 250), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (25, 250, 250), t)
        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fps_Text = "FPS:"+str(int(fps))
        cv2.putText(img, fps_Text, (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (25, 25, 250), 1)
        cv2.imshow("Frame", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


"""
cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (250, 250, 10), 2)
            text=str(int(detection.score[0] * 100))+" >>" 
            cv2.putText(
                img,
                text,
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,2,
                (0, 250, 200),
                2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
"""
