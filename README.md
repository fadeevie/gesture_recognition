# Gesture Recognition Project

Этот проект посвящён распознаванию жестов рук с использованием трёх подходов: Mediapipe + KNN, HOG + SVM и CNN. Мы сравниваем их точность и производительность на датасете LeapGestRecog, а также проверяем работу в реальном времени через веб-камеру.

## О проекте
Цель — изучить, как разные методы машинного обучения справляются с задачей распознавания жестов. В проекте используются:
- **Mediapipe + KNN**: Быстрый и точный метод, основанный на ключевых точках руки.
- **HOG + SVM**: Метод, использующий гистограммы градиентов, быстрый на веб-камере.
- **CNN**: Нейронная сеть, которая требует больше ресурсов, но может быть точнее при правильной настройке.

## Требования
Для работы проекта нужны Python 3.8+ и следующие библиотеки:
- opencv-python
- mediapipe
- scikit-learn
- numpy
- tensorflow
- matplotlib
- seaborn
- tqdm
- scikit-image

Все зависимости перечислены в файле `requirements.txt`.

## Датасет
Проект использует датасет [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data). 
1. Скачайте его с Kaggle.
2. Распакуйте архив в папку `leapGestRecog` рядом с проектом (в корне репозитория).


## Установка
1. Клонируйте репозиторий с GitHub:
   ```bash
   git clone https://github.com/yourusername/gesture_recognition.git
Замените yourusername на ваш логин GitHub.

2. Перейдите в папку проекта:
   ```bash
   cd gesture_recognition
   
3. Установите зависимости:
    ```bash
   pip install -r requirements.txt

## Как запустить
Каждый подход состоит из двух скриптов: обучение модели и тестирование на веб-камере.
1. Mediapipe + KNN
Для запуска данного подхода следует запустить python mediapipe/test_mediapipe.py (используя knn_model.pkl)

2. HOG + SVM
Для запуска данного подхода следует запустить python hog_svm/test_hog_svm.py (используя svm_model.pkl)

3. CNN
Для запуска данного подхода следует запустить python cnn/test_mcnn.py (используя cnn_model_final.h5)
