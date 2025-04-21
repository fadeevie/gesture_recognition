import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Путь к датасету
dataset_path = r"C:\fa\gesture\leapGestRecog"
sessions = [f"{i:02d}" for i in range(10)]
gestures = ['01_palm', '03_fist', '06_index', '05_thumb', '07_ok']
IMG_SIZE = 64

# Загрузка данных
X, y = [], []
class_counts = {gesture: 0 for gesture in gestures}
total_images = 0
for session in sessions:
    for gesture in gestures:
        gesture_path = os.path.join(dataset_path, session, gesture)
        if os.path.exists(gesture_path):
            total_images += len(os.listdir(gesture_path))

print(f"Всего изображений для обработки: {total_images}")

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
                X.append(img_resized)
                y.append(label_idx)
                class_counts[gesture] += 1
                pbar.update(1)

# Проверка данных
print(f"Извлечено {len(X)} примеров. Распределение классов: {class_counts}")

# Преобразование и нормализация
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Проверка входных данных
print(f"Размер X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Размер X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Пример меток y_train: {y_train[:10]}")

# Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(gestures), activation='softmax')
])

# Компиляция с меньшим learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Обучение
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=1)

# Оценка
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Предсказания для метрик
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=gestures))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=gestures, yticklabels=gestures)
plt.title("Confusion Matrix (CNN)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_cnn.png")
plt.close()

# Сохранение модели
model.save("cnn_model_final.h5")
print(f"Модель обучена на {len(X_train)} примерах, протестирована на {len(X_test)} примерах и сохранена!")