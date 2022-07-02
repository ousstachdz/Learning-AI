from unicodedata import name
import cv2
import mediapipe as mp
import time

from pip import main

class HandDetector():
    def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands =  max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(static_image_mode,
                                max_num_hands,
                                model_complexity,
                                min_detection_confidence,
                                min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw :
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                lmList.append((cx, cy, id))
                if id == 12:
                    cv2.circle(img,(cx,cy),20,(0,0,255),cv2.FILLED)
        return lmList



def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHand(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main() 