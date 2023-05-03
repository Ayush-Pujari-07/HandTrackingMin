import cv2
import mediapipe as mp
import time

"""
This is a Python script that uses the mediapipe and OpenCV libraries to detect hands in a video stream. 
The script defines a HandDetector class that takes in several parameters like mode, maxHands, detectionCon, and trackCon. 
These parameters configure the behavior of the hand detection algorithm.
"""


class HandDetector():
    """
    The findHands() method of the HandDetector class takes in an image as input and returns the same image with the detected hand landmarks and connections drawn on it. 
    The script reads frames from the video stream using cv2.VideoCapture(), and passes each frame through the HandDetector to detect hands in it. 
    The detected landmarks are then drawn on the frame, and the frame is displayed using cv2.imshow().

    """

    def __init__(self, mode=False, maxHands=2, modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        # this is because the class take only RGB values
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # THIS IS GETTING CONNECTIONS BETWEEN THE NODES IN THE HAND
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # p rint(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (250, 0, 250), cv2.FILLED) # cx is for x axis position and cy is for y axis position 

        return lmList

# This is a dummy code and could be used for other projects 
def main():
    """
    The script also calculates the frames per second (fps) of the video stream and displays it on the screen using cv2.putText(). 
    The script uses a while loop to keep reading frames from the video stream until the user exits the program by pressing the q key.
    """
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Here 1 denote webcam number '1'
    detector = HandDetector()
    while True:
        # Read the image.
        success, img = cap.read()
        # check hands
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # For showing FPS.
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 3)  # This will show the fps
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    # calling the main function here
    main()
 