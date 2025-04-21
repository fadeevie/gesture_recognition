import cv2
import numpy as np
from skimage.feature import hog
import pickle

# Загрузка модели
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Названия жестов
gestures = ['Palm', 'Fist', 'Index', 'Thumb', 'OK']
IMG_SIZE = 64

# Захват с веб-камеры
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка кадра
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    features = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features = features.reshape(1, -1)

    # Предсказание
    pred = svm.predict(features)[0]
    gesture_name = gestures[pred]

    # Вывод результата
    cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition (HOG + SVM)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()