import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self,
                 mode=False,
                 max_num_hands=2,
                 min_detection_confidence=.5,
                 min_tracking_confidence=.5):
        self.mode = mode
        self.max_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.max_hands,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw = True):
        height, width, channel = img.shape
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_result = self.hands.process(imageRGB)
        hand_landmarks = []
        if hand_result.multi_hand_landmarks:
            for i, hand in enumerate(hand_result.multi_hand_landmarks):
                landmarks = {}
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
                for landmark_id, position in enumerate(hand.landmark):
                    x, y = int(position.x * width), int(position.y * height)
                    landmarks[landmark_id] = (x, y)
                hand_landmarks.append(landmarks)
        return [img, hand_landmarks]

def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    previousTime = 0
    currentTime = 0
    while True:
        success, frame = cap.read()
        if success:
            res = detector.find_hands(frame)
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
            cv2.imshow("image", frame)
            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()