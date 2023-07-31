import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(
max_num_hands=1,
min_detection_confidence=0.8, 
min_tracking_confidence=0.5)

model = load_model('mp_hand_gesture')

f = open('gesture.names')
classNames = f.read().split('\n')
f.close()
print(classNames)

video = cv2.VideoCapture(1)

while True:
    check, frame = video.read()
    x, y, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            prediction = model.predict([landmarks])

            classID = np.argmax(prediction)
            className = classNames[classID].capitalize()

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
