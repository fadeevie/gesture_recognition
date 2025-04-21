import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Путь к датасету
dataset_path = r"C:\fa\gesture\leapGestRecog"
sessions = [f"{i:02d}" for i in range(10)]  # 00–09
gestures = ['01_palm', '03_fist', '06_index', '05_thumb', '07_ok']
X, y = [], []

# Подсчёт общего количества изображений для прогресс-бара
total_images = 0
for session in sessions:
    for gesture in gestures:
        gesture_path = os.path.join(dataset_path, session, gesture)
        if os.path.exists(gesture_path):
            total_images += len(os.listdir(gesture_path))

print(f"Всего изображений для обработки: {total_images}")

# Извлечение признаков с прогресс-баром
with tqdm(total=total_images, desc="Обработка изображений") as pbar:
    for session in sessions:
        for label_idx, gesture in enumerate(gestures):
            gesture_path = os.path.join(dataset_path, session, gesture)
            if not os.path.exists(gesture_path):
                print(f"Папка {gesture_path} не найдена, пропускаем.")
                continue
            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось загрузить {img_path}, пропускаем.")
                    pbar.update(1)
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Обработка изображения Mediapipe
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
                    X.append(features)
                    y.append(label_idx)
                else:
                    print(f"Не удалось извлечь ключевые точки из {img_path}")
                pbar.update(1)

# Проверка, что данные собраны
if not X:
    raise ValueError("Не удалось извлечь признаки из датасета. Проверьте путь и наличие изображений.")

print(f"Извлечено {len(X)} примеров с ключевыми точками.")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = knn.predict(X_test)

# Вычисление метрик
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=gestures))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=gestures, yticklabels=gestures)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
print("Матрица ошибок сохранена как 'confusion_matrix.png'")

# Сохранение модели
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

hands.close()
print(f"Модель обучена на {len(X_train)} примерах, протестирована на {len(X_test)} примерах и сохранена!")