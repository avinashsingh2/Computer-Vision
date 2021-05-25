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
        self.fingersTip = [4, 8, 12, 16, 20]
        self.hand_result = None

    def find_hands(self, img, draw = True):
        height, width, channel = img.shape
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hand_result = self.hands.process(imageRGB)
        if self.hand_result.multi_hand_landmarks:
            for i, hand in enumerate(self.hand_result.multi_hand_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def get_landmarks_position(self,img, landmark_list = None, draw = False):
        height, width, channel = img.shape
        hand_landmarks = {"Left":{}, "Right":{}}
        if self.hand_result.multi_hand_landmarks:
            for hand_type, hand in zip(self.hand_result.multi_handedness, self.hand_result.multi_hand_landmarks):
                hand_type = hand_type.classification[0].label
                landmarks = {}
                for landmark in range(21):
                    landmark_pos = hand.landmark[landmark]
                    landmarks[landmark] = (int(landmark_pos.x*width), int(landmark_pos.y*height))
                    if draw:
                        if hand_type == "Right":
                            cv2.circle(img,landmarks[landmark],15,(0,0,255),cv2.FILLED)
                        else:
                            cv2.circle(img, landmarks[landmark], 15, (0, 255, 255), cv2.FILLED)

                hand_landmarks[hand_type]=landmarks
        return hand_landmarks

    def get_fingers_up(self, landmark_position):
        landmark_list = self.fingersTip
        fingers_up_list = {"Left":[0,0,0,0,0],"Right":[0,0,0,0,0]}
        for hand_type, landmarks in landmark_position.items():
            #thumb
            if landmarks:
                if landmarks.get(landmark_list[0]):
                    if hand_type == 'Left':
                        if landmarks[landmark_list[0]][0] > landmarks[landmark_list[0]-1][0]:
                            fingers_up_list[hand_type][0] = 1
                    else:
                        if landmarks[landmark_list[0]][0] < landmarks[landmark_list[0]-1][0]:
                            fingers_up_list[hand_type][0] = 1
                for i in range(1,5):
                    if landmarks.get(landmark_list[i]):
                        if landmarks[landmark_list[i]][1] < landmarks[landmark_list[i]-2][1]:
                            fingers_up_list[hand_type][i]=1
        return fingers_up_list

    def count_finger(self,img, fingers_up_list):
        finger_sum = sum(fingers_up_list["Left"]) + sum(fingers_up_list["Right"])
        cv2.putText(img,f'sum = {finger_sum}',(10,80),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
        return sum



def main():
    detector = HandDetector(min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    previousTime = 0
    currentTime = 0
    while True:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            image = detector.find_hands(frame)
            position_list = detector.get_landmarks_position(frame, detector.fingersTip)
            fingers_up = detector.get_fingers_up(position_list)
            sum = detector.count_finger(image,fingers_up)
            print(fingers_up)
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(frame, f'FPS:{int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("image", frame)
            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()