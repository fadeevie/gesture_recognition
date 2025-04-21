import cv2
import mediapipe as mp
import numpy as np
import pickle

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Загрузка модели
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# Названия жестов
gestures = ['Palm', 'Fist', 'Index', 'Thumb', 'OK']

# Захват с веб-камеры
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Отрисовка ключевых точек
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Извлечение признаков
            features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            features = features.reshape(1, -1)

            # Предсказание
            pred = knn.predict(features)[0]
            gesture_name = gestures[pred]

            # Вывод результата
            cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()