# Архитектура системы классификации пластика

## Общая структура проекта

Проект представляет собой веб-приложение для классификации типов пластика с помощью нейронной сети. Система состоит из двух основных компонентов:

1. **Модуль обучения модели** (`plastic_type_classifier.py`)
2. **Веб-приложение** (`app.py`)

## Архитектура приложения

```
Клиент (Браузер)
index.html + Tailwind CSS + JavaScript             
- Загрузка изображений                             
- Предпросмотр                       
- Отображение результатов
                            │
                            │ HTTP/HTTPS
                            ▼
Flask Web Server (app.py) 

Роуты:                                             
- GET  /            → index.html
- POST /predict     → JSON результат
                            │                                  
                            ▼                               
Обработка изображений:
- PIL (открытие и конвертация)
- Resize до 224×224 
- Нормализация (ResNet50 preprocess)
                            │                                  
                            ▼                                  
TensorFlow Model (ResNet-50)
- Загрузка модели из .keras файла
- Предсказание класса
- 6 классов: PET, PE, PC, PP, PS, Others
```

## Компоненты системы

### 1. Frontend (Клиентская часть)

**Технологии:**
- HTML5
- Tailwind CSS (через CDN)
- Vanilla JavaScript

**Основные функции:**
- Загрузка изображений через `<input type="file">`
- Предпросмотр загруженного изображения
- Отправка изображения на сервер через Fetch API
- Отображение результатов с анимацией загрузки
- Адаптивный дизайн

**Структура `templates/index.html`:**
```
- Header (заголовок и описание)
- Upload section (кнопка выбора файла)
- Preview container (предпросмотр)
- Predict button (кнопка классификации)
- Loading indicator (индикатор загрузки)
- Result container (результат с уверенностью)
- Error container (обработка ошибок)
```

### 2. Backend (Серверная часть)

**Фреймворк:** Flask (Python)

**Основные модули:**

#### `app.py` - Веб-сервер

**Зависимости:**
- `flask` - веб-фреймворк
- `tensorflow` - запуск модели
- `numpy` - работа с массивами
- `PIL (Pillow)` - обработка изображений

**Ключевые функции:**

1. **`load_model()`**
   - Загружает обученную модель при старте приложения
   - Использует глобальную переменную для кэширования
   - Оптимизация: модель загружается один раз

2. **`index()`** (GET `/`)
   - Возвращает главную страницу `index.html`
   - Рендеринг через Jinja2

3. **`predict()`** (POST `/predict`)
   - Принимает изображение через multipart/form-data
   - Валидация файла
   - Предобработка изображения:
     - Конвертация в RGB
     - Изменение размера до 224×224
     - Преобразование в numpy array
     - Нормализация через `resnet50.preprocess_input`
   - Предсказание через модель
   - Возврат JSON с классом и уверенностью

**Конфигурация:**
```python
CLASS_NAMES = ['PET', 'PE', 'PC', 'PP', 'PS', 'Others']
IMG_SIZE = (224, 224)
MODEL_PATH = 'plastic_type_classifier.keras'
```

### 3. Модель машинного обучения

**Архитектура:** ResNet-50 (Transfer Learning)

**Структура модели:**

```
Input Layer (224×224×3)
         │
         ▼
┌────────────────────────┐
│  ResNet-50 Base        │
│  (weights=None)        │
│  - 50 layers           │
│  - trainable=True      │
│  - include_top=False   │
└────────────────────────┘
         │
         ▼
GlobalAveragePooling2D
         │
         ▼
BatchNormalization
         │
         ▼
Dropout (0.5)
         │
         ▼
Dense (256 units, relu)
│  kernel_regularizer=l2(1e-4)
         │
         ▼
BatchNormalization
         │
         ▼
Dropout (0.5)
         │
         ▼
Dense (6 units, softmax)
         │
         ▼
Output: [PET, PE, PC, PP, PS, Others]
```

**Параметры обучения:**

```python
# Оптимизация
Optimizer: Adam (learning_rate=1e-3)
Loss: categorical_crossentropy
Metrics: accuracy

# Callbacks
- EarlyStopping (patience=8, monitor='val_accuracy')
- ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
- ModelCheckpoint (save_best_only=True)

# Данные
Batch Size: 16
Epochs: 40 (max)
Validation Split: 20%
Image Size: 224×224
```

**Аугментация данных:**
```python
- rotation_range: 30°
- width_shift_range: 0.2
- height_shift_range: 0.2
- shear_range: 0.15
- zoom_range: 0.2
- horizontal_flip: True
- brightness_range: [0.7, 1.3]
```

### 4. Обучение модели (`plastic_type_classifier.py`)

**Процесс обучения:**

1. **Загрузка данных**
   - Чтение из `plastic_types_dataset/`
   - 6 папок по классам
   - Автоматическое разделение на train/val (80/20)

2. **Предобработка**
   - ResNet50 preprocessing function
   - Применение аугментаций
   - Batch generation через DataGenerator

3. **Построение модели**
   - Создание базовой ResNet-50
   - Добавление custom head
   - Компиляция с Adam optimizer

4. **Обучение**
   - Training loop с callbacks
   - Early stopping для предотвращения переобучения
   - Сохранение лучшей модели

5. **Визуализация**
   - Графики accuracy (train/val)
   - Графики loss (train/val)
   - Сохранение в .keras формат

## Поток данных

### 1. Обучение модели

```
Dataset (папки с изображениями)
         │
         ▼
ImageDataGenerator
         │
         ├── Train Generator (80%)
         │       │
         │       ├── Аугментация
         │       └── Batch generation
         │
         └── Validation Generator (20%)
                 │
                 └── Без аугментации
         
         ▼
ResNet-50 Model
         │
         ├── Forward Pass
         ├── Loss Calculation
         └── Backpropagation
         
         ▼
Callbacks
         │
         ├── EarlyStopping
         ├── ReduceLROnPlateau
         └── ModelCheckpoint
         
         ▼
Saved Model (.keras)
```

### 2. Инференс (Предсказание)

```
Client Browser
         │
         │ POST /predict (image file)
         ▼
Flask Server
         │
         ├── Validate file
         ├── PIL: Open & Convert to RGB
         ├── Resize to 224×224
         ├── Numpy conversion
         └── ResNet50 preprocessing
         
         ▼
Loaded Model
         │
         ├── model.predict()
         ├── np.argmax() → class index
         └── np.max() → confidence
         
         ▼
JSON Response
{
  "class": "PET",
  "confidence": 0.85
}
         │
         ▼
Client Display
```

## Технологии и зависимости

### Backend
- **Python 3.8+**
- **TensorFlow 2.x** - Deep Learning framework
- **Flask 2.x** - Web framework
- **NumPy** - Numerical operations
- **Pillow (PIL)** - Image processing
- **Matplotlib** - Visualization (для обучения)

### Frontend
- **HTML5** - Структура
- **Tailwind CSS 3.x** - Styling (через CDN)
- **JavaScript (ES6+)** - Логика клиента
- **Fetch API** - HTTP запросы

### Инструменты разработки
- **Git** - Version control
- **VS Code / PyCharm** - IDE

## Структура файлов

```
plastic_classifier/
│
├── README.md                          
├── LICENSE                            
├── .gitignore                         
├── requirements.txt                   
├── CONTRIBUTING.md                    
├── CHANGELOG.md                       
│
├── app.py                          # Flask сервер
├── plastic_type_classifier.py      # Скрипт обучения
├── plastic_type_classifier.keras   # Обученная модель
├── requirements.txt                # Python зависимости
│
├── templates/
│   └── index.html                      # HTML шаблон
│
├── plastic_types_dataset/          # Датасет
│   ├── PET/
│   ├── PE/
│   ├── PC/
│   ├── PP/
│   ├── PS/
│   └── Others/
│
├── docs/
│   ├── architecture.md             # Этот файл
│   └── screenshots/                # Скриншоты
│       ├── main_page.png
│       └── result.png
│
└── media/
    ├── video_overview.md           # Ссылки на видео
    └── demo.gif                    # GIF-демонстрация
```

## Безопасность и оптимизация

### Безопасность
- Валидация загружаемых файлов
- Обработка исключений при загрузке
- Ограничение типов файлов (изображения)
- CORS настройка (при необходимости)

### Оптимизация
- **Кэширование модели** - загрузка один раз при старте
- **Batch processing** - эффективная обработка
- **Image resizing** - уменьшение размера до 224×224
- **GPU support** - TensorFlow автоматически использует GPU при наличии

### Масштабируемость
- Stateless архитектура (можно масштабировать горизонтально)
- Возможность вынесения модели в отдельный микросервис
- Поддержка load balancing

## Развертывание

### Локальная разработка
```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск обучения
python plastic_type_classifier.py

# Запуск сервера
python app.py
```

### Production
1. **WSGI Server**: Gunicorn или uWSGI
2. **Reverse Proxy**: Nginx
3. **Containerization**: Docker
4. **Cloud**: Heroku, AWS, Google Cloud

### Конфигурация production
```python
# Отключение debug mode
app.run(debug=False, host='0.0.0.0', port=5000)

# Использование gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Мониторинг и логирование

### Логирование
- Загрузка модели при старте
- Ошибки обработки изображений
- HTTP запросы (в debug mode)

### Метрики
- Точность модели (val_accuracy)
- Время отклика API
- Количество запросов

## Возможные улучшения

1. **Модель**
   - Использование предобученных весов (ImageNet)
   - Fine-tuning вместо обучения с нуля
   - Ensemble моделей

2. **Производительность**
   - Кэширование предсказаний
   - Асинхронная обработка
   - Оптимизация модели (TensorRT, quantization)

3. **Функциональность**
   - Поддержка множественных изображений
   - История предсказаний
   - API документация (Swagger/OpenAPI)
   - Аутентификация пользователей

4. **DevOps**
   - CI/CD pipeline
   - Автоматическое тестирование
   - Контейнеризация (Docker)
   - Monitoring (Prometheus, Grafana)

## Заключение

Система представляет собой полнофункциональное веб-приложение для классификации пластика, использующее современные технологии глубокого обучения и веб-разработки. Архитектура обеспечивает хорошую производительность, масштабируемость и простоту поддержки.