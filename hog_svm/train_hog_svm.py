import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Путь к датасету
dataset_path = r"C:\fa\gesture\leapGestRecog"
sessions = [f"{i:02d}" for i in range(10)]
gestures = ['01_palm', '03_fist', '06_index', '05_thumb', '07_ok']
IMG_SIZE = 64
X, y = [], []

# Подсчёт общего количества изображений
total_images = 0
for session in sessions:
    for gesture in gestures:
        gesture_path = os.path.join(dataset_path, session, gesture)
        if os.path.exists(gesture_path):
            total_images += len(os.listdir(gesture_path))

print(f"Всего изображений для обработки: {total_images}")

# Извлечение HOG-признаков
class_counts = {gesture: 0 for gesture in gestures}
with tqdm(total=total_images, desc="Обработка изображений") as pbar:
    for session in sessions:
        for label_idx, gesture in enumerate(gestures):
            gesture_path = os.path.join(dataset_path, session, gesture)
            if not os.path.exists(gesture_path):
                continue
            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    pbar.update(1)
                    continue
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                features = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
                X.append(features)
                y.append(label_idx)
                class_counts[gesture] += 1
                pbar.update(1)

print(f"Извлечено {len(X)} примеров. Распределение классов: {class_counts}")

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Предсказание и метрики
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=gestures))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=gestures, yticklabels=gestures)
plt.title("Confusion Matrix (HOG + SVM)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_hog_svm.png")

# Сохранение модели
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print(f"Модель обучена на {len(X_train)} примерах, протестирована на {len(X_test)} примерах и сохранена!")