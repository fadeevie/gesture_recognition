import cv2
import numpy as np
import tensorflow as tf

# Загрузка модели
model = tf.keras.models.load_model("cnn_model_final.h5")

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
    # Увеличение контрастности
    gray = cv2.equalizeHist(gray)
    # Удаление фона (примерная маска по порогу)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    gray = cv2.bitwise_and(gray, gray, mask=thresh)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    input_data = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Предсказание
    pred = model.predict(input_data, verbose=0)
    gesture_idx = np.argmax(pred)
    gesture_name = gestures[gesture_idx]

    # Вывод результата
    cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition (CNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()