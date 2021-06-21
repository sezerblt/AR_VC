import cv2
import mediapipe
import time

class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectConf=0.5,trackConf=0.5):
        self.mode=mode
        self.maxHands = maxHands
        self.detectCon=detectConf
        self.trackCon=trackConf
        
        self.mpHands = mediapipe.solutions.hands 
        self.mpDraw =mediapipe.solutions.drawing_utils
        
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.detectCon,
            self.trackCon
            )
        self.results=None
        
    def findHands(self,frame,drawMode=True):
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if drawMode:
                    self.mpDraw.draw_landmarks(
                        frame,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS
                    )
                    
        return frame

    def findPosition(self,frame,handNo=0,drawMode=True):
        landMarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for mark_id,mark in enumerate(myHand.landmark):
                h,w,c = frame.shape
                cx,cy=int(mark.x+w),int(mark.y+h)
                landMarkList.append([mark_id,cx,cy])
                if drawMode:
                    cv2.circle(frame,(cx,cy),10,(250,0,250),cv2.FILLED)
                
        return landMarkList
                 
def test_start():
    pTime=0
    cTime=0
    capture=cv2.VideoCapture(0)
    detector=HandDetector()
    while True:
        isOk,frame=capture.read()
        frame=cv2.flip(frame,1)

        frame = detector.findHands(frame)
        listOfMArks = detector.findPosition(frame)

        if len(listOfMArks)!=0:
           print(">> ",listOfMArks[4])
        
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        
        cv2.putText(frame,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(0,50,250),3)
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    

if __name__=="__main__":
    test_start()
