# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

## Общее описание

**Задача:** Разработать приложение для классификации изображений трех достопримечательностей Нижнего Новгорода:
- Нижегородский Кремль
- Дворец труда
- Архангельский собор

**Реализованные алгоритмы:**
1. **Bag of Visual Words** - мешок слов
2. **Convolutional Neural Network** - подход с нейросетями

## Архитектура проекта

### Структура проекта:
```text
lab3/
├── scripts/
│   ├── main.py                 # точка входа
│   ├── data_loader.py          # подготовка данных
│   ├── base_classifier.py      # абстрактный класс
│   ├── bow_classifier.py       
│   └── cnn_classifier.py       
├── models/                     
├── my_images/                  
├── NNClassification/           
├── requirements.txt          
└── README.md                   
```

## Алгоритм "Мешок слов"

### Математическая основа:

#### 1. Извлечение признаков (SIFT/ORB)
Для каждого изображения $I$ находим набор дескрипторов:
$$D = \{d_1, d_2, ..., d_n\}, \quad d_i \in \mathbb{R}^{128} \text{ (для SIFT)}$$

**Дескриптор SIFT** - 128-мерный вектор, описывающий локальную область вокруг ключевой точки. Вычисляется с помощью гистограммы градиентов в 16×16 окрестности точки.

#### 2. Построение визуального словаря (K-Means)
Объединяем все дескрипторы со всех обучающих изображений:
$$D_{\text{all}} = \bigcup_{j=1}^{N} D^{(j)}$$

Применяем K-Means кластеризацию для нахождения $K$ визуальных слов:
$$\min_{C} \sum_{i=1}^{|D_{\text{all}}|} \min_{j=1}^{K} \|d_i - c_j\|^2$$
где $C = \{c_1, c_2, ..., c_K\}$ - словарь.

#### 3. Преобразование изображения в гистограмму
Для изображения с дескрипторами $D = \{d_1, ..., d_n\}$ вычисляем гистограмму:
$$h_j = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(\text{argmin}_k \|d_i - c_k\| = j)$$
где $\mathbb{I}$ - индикаторная функция, $h_j$ - частота $j$-го визуального слова.

#### 4. Классификация
Обучаем SVM на гистограммах:
$$f(h) = \text{sign}(w^T h + b)$$.

### Реализация в коде:

```python
1. extract_descriptors() - SIFT/ORB дескрипторы через cv2.SIFT_create()
2. build_vocabulary() - K-Means кластеризация через sklearn.cluster.KMeans
3. image_to_histogram() - преобразование через kmeans.predict() и np.histogram()
4. _train_impl() - обучение SVM через sklearn.svm.SVC
```

### Параметры BoW:
- `--k`: размер словаря
- `--detector`: детектор признаков (sift/orb)

## Сверточная нейронная сеть (CNN)

### Математическая основа:

#### 1. Transfer Learning
Используем предобученную VGG16 на ImageNet. Замораживаем веса сверточных слоев:
$$W_{\text{conv}} \leftarrow W_{\text{ImageNet}}, \quad \frac{\partial L}{\partial W_{\text{conv}}} = 0$$

Обучение только новых слоев:
$$W_{\text{fc}} \leftarrow \text{random}, \quad \frac{\partial L}{\partial W_{\text{fc}}} \neq 0$$

#### 3. Функция потерь (cross-entropy):
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{3} y_{i,c} \log(\hat{y}_{i,c})$$
где $y$ - кодирование истинных меток, $\hat{y}$ - предсказанные вероятности.

#### 4. Оптимизатор Adam:
Обновление весов по формуле:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
где $\hat{m}_t$ и $\hat{v}_t$ - оценки первого и второго моментов градиентов.

### Реализация в коде:
```python
1. create_cnn_model() - VGG16(weights='imagenet', include_top=False)
2. preprocess_image() - resize(224,224), /255.0, BGR→RGB
3. _train_impl() - model.fit() с EarlyStopping и ReduceLROnPlateau
4. _test_impl() - model.predict() и вычисление метрик
```

### Параметры CNN:
- `--epochs`: количество эпох обучения
- `--batch_size`: размер батча
- `--lr`: скорость обучения оптимизатора

## Структура данных

### Исходный датасет:
```text
NNClassification/
├── NNSUDataset/                   
│   ├── 01_NizhnyNovgorodKremlin/
│   ├── 04_ArkhangelskCathedral/
│   └── 08_PalaceOfLabor/
├── ExtDataset/                    
│   ├── 01_NizhnyNovgorodKremlin/
│   ├── 04_ArkhangelskCathedral/
│   ├── 08_PalaceOfLabor/
│   └── reference                  
└── train_test_split/
    ├── train.txt              
    └── test.txt                  
```

### Дополнительные изображения:
```text
my_images/
├── NizhnyNovgorodKremlin/         
├── ArchangelCathedral/         
├── PalaceOfLabor/                
└── reference                
```

### 2. Основные команды:

#### Bag of Visual Words:
```bash
# Обучение и тестирование
python scripts/main.py --algo bow --detector sift --k 200 --mode both
```

```bash
# Только обучение
python scripts/main.py --algo bow --mode train
```

```bash
# Только тестирование 
python scripts/main.py --algo bow --mode test
```

```bash
# Визуализация ключевых точек
python scripts/main.py --mode visualize --detector sift --image_path "путь/к/изображению.jpg"
```

#### CNN:
```bash
# Обучение и тестирование
python scripts/main.py --algo cnn --epochs 10 --batch_size 8 --mode both
```

```bash
# Тестирование на одном изображении
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

### Метрики качества:

#### 1. Accuracy: доля правильных предсказаний
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### 2. Precision: точность для каждого класса
$$\text{Precision} = \frac{TP}{TP + FP}$$

#### 3. Recall: полнота для каждого класса
$$\text{Recall} = \frac{TP}{TP + FN}$$

#### 4. F1-score: гармоническое среднее precision и recall
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
