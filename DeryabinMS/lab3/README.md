# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

## Содержание
1. [Общее описание](#общее-описание)
2. [Архитектура проекта](#архитектура-проекта)
3. [Алгоритм "Мешок визуальных слов" (BoW)](#алгоритм-мешок-визуальных-слов-bow)
4. [Сверточная нейронная сеть (CNN) с Transfer Learning](#сверточная-нейронная-сеть-cnn-с-transfer-learning)
5. [Структура данных](#структура-данных)
6. [Установка и запуск](#установка-и-запуск)
7. [Результаты](#результаты)
8. [Ответы на возможные вопросы](#ответы-на-возможные-вопросы)

## Общее описание

**Задача:** Разработать приложение для классификации изображений трех достопримечательностей Нижнего Новгорода:
- Нижегородский Кремль
- Дворец труда
- Архангельский собор

**Реализованные алгоритмы:**
1. **Bag of Visual Words (BoW)** - классический подход компьютерного зрения
2. **Convolutional Neural Network (CNN)** с Transfer Learning - современный подход с нейросетями

**Требования задания выполнены:**
- ✓ Модульная объектно-ориентированная архитектура
- ✓ Автоматическое формирование тестовой выборки
- ✓ Добавление дополнительных изображений
- ✓ Визуализация ключевых точек для BoW
- ✓ Подробная оценка качества (accuracy, confusion matrix, отчет)
- ✓ Сохранение и загрузка моделей
- ✓ Файл с ссылками на источники изображений

## Архитектура проекта

### Структура файлов:
practial_work_3/
├── scripts/
│   ├── main.py                 # Точка входа, парсинг аргументов
│   ├── data_loader.py          # Загрузка и подготовка данных
│   ├── base_classifier.py      # Базовый абстрактный класс
│   ├── bow_classifier.py       # Реализация BoW
│   └── cnn_classifier.py       # Реализация CNN
├── models/                     # Сохраненные модели
├── my_images/                  # Дополнительные изображения
├── NNClassification/           # Исходный датасет
├── requirements.txt            # Зависимости Python
└── README.md                   # Документация

### Принцип ООП в проекте:
1. **Абстракция:** `BaseClassifier` определяет общий интерфейс
2. **Наследование:** `BOWClassifier` и `CNNClassifier` наследуют общий функционал
3. **Полиморфизм:** вызов `train_from_items()` работает для обоих классификаторов
4. **Инкапсуляция:** внутренняя реализация каждого алгоритма скрыта

## Алгоритм "Мешок визуальных слов" (BoW)

### Математическая основа:

#### 1. Извлечение признаков (SIFT/ORB)
Для каждого изображения $I$ находим набор дескрипторов:
$$D = \{d_1, d_2, ..., d_n\}, \quad d_i \in \mathbb{R}^{128} \text{ (для SIFT)}$$

**Дескриптор SIFT** - 128-мерный вектор, описывающий локальную область вокруг ключевой точки. Вычисляется с помощью гистограммы градиентов в 16×16 окрестности точки.

#### 2. Построение визуального словаря (K-Means)
Объединяем все дескрипторы со всех обучающих изображений:
$$D_{\text{all}} = \bigcup_{j=1}^{N} D^{(j)}$$

Применяем K-Means кластеризацию для нахождения $K$ центроидов (визуальных слов):
$$\min_{C} \sum_{i=1}^{|D_{\text{all}}|} \min_{j=1}^{K} \|d_i - c_j\|^2$$
где $C = \{c_1, c_2, ..., c_K\}$ - центроиды (словарь).

#### 3. Преобразование изображения в гистограмму
Для изображения с дескрипторами $D = \{d_1, ..., d_n\}$ вычисляем гистограмму:
$$h_j = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(\text{argmin}_k \|d_i - c_k\| = j)$$
где $\mathbb{I}$ - индикаторная функция, $h_j$ - частота $j$-го визуального слова.

#### 4. Классификация (SVM)
Обучаем SVM на гистограммах:
$$f(h) = \text{sign}(w^T h + b)$$
Для многоклассовой классификации используем стратегию "one-vs-rest".

### Реализация в коде:

```python
# Основные шаги в bow_classifier.py:
1. extract_descriptors() - SIFT/ORB дескрипторы через cv2.SIFT_create()
2. build_vocabulary() - K-Means кластеризация через sklearn.cluster.KMeans
3. image_to_histogram() - преобразование через kmeans.predict() и np.histogram()
4. _train_impl() - обучение SVM через sklearn.svm.SVC
```

### Параметры BoW:
- `--k`: размер словаря (количество кластеров K-Means)
- `--detector`: детектор признаков (sift/orb)
- **SIFT**: 128-мерные дескрипторы, инвариантен к масштабу и повороту
- **ORB**: 32-мерные бинарные дескрипторы, быстрее, но менее точный

## Сверточная нейронная сеть (CNN) с Transfer Learning

### Математическая основа:

#### 1. Transfer Learning
Используем предобученную VGG16 на ImageNet. Замораживаем веса сверточных слоев:
$$
W_{\text{conv}} \leftarrow W_{\text{ImageNet}}, \quad \frac{\partial L}{\partial W_{\text{conv}}} = 0
$$
Обучение только новых полносвязных слоев:
$$
W_{\text{fc}} \leftarrow \text{random}, \quad \frac{\partial L}{\partial W_{\text{fc}}} \neq 0
$$

#### 2. Архитектура VGG16:
Input (224×224×3)
↓
Conv3-64 → Conv3-64 → MaxPool
↓
Conv3-128 → Conv3-128 → MaxPool
↓
Conv3-256 → Conv3-256 → Conv3-256 → MaxPool
↓
Conv3-512 → Conv3-512 → Conv3-512 → MaxPool
↓
Conv3-512 → Conv3-512 → Conv3-512 → MaxPool
↓
GlobalAveragePooling2D()
↓
Dense(256, ReLU) → Dropout(0.5) → BatchNorm
↓
Dense(128, ReLU) → Dropout(0.25) → BatchNorm
↓
Dense(3, softmax)

#### 3. Функция потерь (categorical cross-entropy):
$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{3} y_{i,c} \log(\hat{y}_{i,c})
$$
где $y$ - one-hot кодирование истинных меток, $\hat{y}$ - предсказанные вероятности.

#### 4. Оптимизатор Adam:
Обновление весов по формуле:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$
где $\hat{m}_t$ и $\hat{v}_t$ - оценки первого и второго моментов градиентов.

### Реализация в коде:
```python
# Основные шаги в cnn_classifier.py:
1. create_cnn_model() - VGG16(weights='imagenet', include_top=False)
2. preprocess_image() - resize(224,224), /255.0, BGR→RGB
3. _train_impl() - model.fit() с EarlyStopping и ReduceLROnPlateau
4. _test_impl() - model.predict() и вычисление метрик
```

### Параметры CNN:
- `--epochs`: количество эпох обучения
- `--batch_size`: размер батча
- `--lr`: скорость обучения оптимизатора Adam

## Структура данных

### Исходный датасет:
NNClassification/
├── NNSUDataset/                    # Фото от студентов
│   ├── 01_NizhnyNovgorodKremlin/
│   ├── 04_ArkhangelskCathedral/
│   └── 08_PalaceOfLabor/
├── ExtDataset/                     # Фото из интернета
│   ├── 01_NizhnyNovgorodKremlin/
│   ├── 04_ArkhangelskCathedral/
│   ├── 08_PalaceOfLabor/
│   └── reference                   # Файл с ссылками на источники
└── train_test_split/
    ├── train.txt                  # 131 изображений для обучения
    └── test.txt                   # Автоматически создается программой

### Ваши дополнительные изображения:
my_images/
├── NizhnyNovgorodKremlin/         # Ваши фото Кремля (английские названия)
├── ArkhangelskCathedral/          # Ваши фото Собора
├── PalaceOfLabor/                 # Ваши фото Дворца
└── my_sources.txt                 # Файл со ссылками в формате "папка/файл,URL"

### Принцип разделения данных:
- **Обучающая выборка:** изображения из train.txt + ваши изображения из my_images/
- **Тестовая выборка:** все остальные изображения из датасета (не входящие в обучающую)
- **Ваши изображения** добавляются ТОЛЬКО в обучающую выборку для улучшения качества модели

## Установка и запуск

### 1. Установка зависимостей:
```bash
pip install -r requirements.txt
```

### 2. Основные команды:

#### Bag of Visual Words:
# Обучение и тестирование BoW с SIFT (200 кластеров)
```bash
python scripts/main.py --algo bow --detector sift --k 200 --mode both
```

# Только обучение
```bash
python scripts/main.py --algo bow --mode train
```

# Только тестирование (загружает сохраненную модель)
```bash
python scripts/main.py --algo bow --mode test
```

# Визуализация ключевых точек SIFT
```bash
python scripts/main.py --mode visualize --detector sift --image_path "путь/к/изображению.jpg"
```

#### CNN с Transfer Learning:
# Обучение и тестирование CNN (10 эпох)
```bash
python scripts/main.py --algo cnn --epochs 10 --batch_size 8 --mode both
```

# Тестирование на одном изображении
```bash
python scripts/main.py --mode single --algo cnn --image_path "my_images/Kremlin/photo.jpg"
```
### 3. Параметры командной строки:
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--data_dir` | Корневая директория с данными | `NNClassification` |
| `--train_list` | Файл с обучающей выборкой | `train_test_split/train.txt` |
| `--extra_dir` | Папка с дополнительными изображениями | `my_images` |
| `--algo` | Алгоритм: `bow` или `cnn` | `bow` |
| `--mode` | Режим: `train`, `test`, `both`, `visualize`, `single` | `both` |
| `--k` | Размер словаря для BoW | `200` |
| `--detector` | Детектор для BoW: `sift`, `orb` | `sift` |
| `--epochs` | Эпохи обучения для CNN | `10` |
| `--batch_size` | Batch size для CNN | `8` |
| `--lr` | Learning rate для CNN | `0.001` |
| `--model_dir` | Директория для моделей | `models` |

## Результаты

### Метрики оценки:
- **Accuracy:** доля правильных предсказаний
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$
- **Precision:** точность для каждого класса
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$
- **Recall:** полнота для каждого класса
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$
- **F1-score:** гармоническое среднее precision и recall
  $$
  F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$
