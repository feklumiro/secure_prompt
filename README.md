# secure-prompt
## Описание
Классификатор для обнаружения **prompt-injection / jailbreak атак** в запросах к языковым моделям (LLM).

Классификация производится с помощью **RandomForestClassifier**, используя следующие признаки:
- Статистические признаки текста (распределение по частям речи, энтропия символов и др.)
- Сигнатурные признаки (количество найденных конструкций, распределение по категориям и др.)
- Семантические признаки (векторное сходство семантики запроса с jailbreak категориями, наиболее близкий, top-k и др.)

## Использование
### Как установить
- Python3 уже должен быть установлен.
- Для изоляции проекта рекомендуется развернуть виртуальное окружение:
```bash
python3 -m venv env
source env/bin/activate
```
- Клонируйте репозиторий с гитхаб:
```bash
git clone https://github.com/feklumiro/secure_prompt.git
cd secure-prompt
```
- Установите зависимости:
```bash
pip install -r requirements.txt
```
### Конфигурация
Создайте в корне проекта файл `.env` и заполните необходимые поля:
```
# Обучающие данные
JAILBREAK_TRAIN_PATH=path/to/jailbreak_train.csv
BENIGN_TRAIN_PATH=path/to/benign_train.csv

# Тестовые данные
JAILBREAK_TEST_PATH=path/to/jailbreak_test.csv
BENIGN_TEST_PATH=path/to/benign_test.csv

# Пороги классификации
THRESHOLD=0.5
THRESHOLD_VECTOR=0.7

# Векторная модель
VECTOR_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```
- JAILBREAK_TRAIN_PATH - путь до файла с обучающими данными (jailbreak)
- BENIGN_TRAIN_PATH - путь до файла с обучающими данными (benign)
- JAILBREAK_TEST_PATH - путь до файла с тестовыми данными (jailbreak)
- BENIGN_TEST_PATH - путь до файла с тестовыми данными (benign)
- THRESHOLD - пороговое значение jailbreak для режима без векторной модели
- THRESHOLD_VECTOR - пороговое значение jailbreak для режима с векторной моделью
- VECTOR_MODEL_NAME - модель эмбеддингов из HuggingFaceHub

### Подготовка данных
Для обучения необходимо собрать два набора данных:
- JAILBREAK (вредоносные запросы)
- BENIGN (безопасные запросы)

Данные должны быть представлены в виде `.csv` файла с единственным столбцом без дополнительной информации.

Данные могут быть собраны из открытых источников, SIEM или TI-решений, или может использоваться набор из данного репозитория, доступный в `/data` (в данном наборе рассмотрены категории `override-based`, `role manipulation` и `system info extraction`).

### Требования к качеству данных
Для достижения высокой точности обучающие данные должны соблюдать следующие требования (для каждого из используемых языков):
- Для каждой категории jailbreak не менее 300 строк обучающих данных.
- Количество строк в benign-данных должно совпадать с количеством строк в jailbreak-данных с точностью до 10%.

При соблюдении описанных условий гарантируется `accuracy >= 0.9` на тестовой выборке с не менее 100 вредоносных запросов каждой представленной категории jailbreak и не менее 100 безопасных запросов.

### Обучение модели
Для обучения модели выполните:
```bash
python3 secure_prompt/ML/train.py
```
или
```python
from secure_prompt.ML.train import train
train()
```
Модели будут обучены на данных из файлов, указанных в `JAILBREAK_TRAIN_PATH` и `BENIGN_TRAIN_PATH` в файле `.env`.

Non-vector модель будет сохранена в `/models/model.pkl`, vector модель будет сохранена в `/models/model_vector.pkl`.

### Тестирование
Для тестирования используются данные из файлов, указанных в `JAILBREAK_TEST_PATH` и `BENIGN_TEST_PATH` из файле `.env`.

Для запуска тестов выполните:
```bash
python3 tests/pipeline_test.py
```

Пороговые значения для моделей могут быть подобраны эмпирически, а также с помощью `threshold.py`:
```bash
python3 threshold.py
```

### Использование
Для использования классификатора импортируйте его из своего Python-приложения:
```python
from secure_prompt.core.decision import DecisionCore
```
Классификатор имеет 2 режима работы:
- `Non-vector` - без векторной модели, только ML и regex признаки
- `Vector` - гибридный режим с векторной моделью

Для инициализации классификатора в режиме `Non-vector` выполните:
```python
classifier = DecisionCore(use_vector=False)
```
Для инициализации классификатора в режиме `Vector` выполните:
```python
classifier = DecisionCore()
```
Также для использования в режиме `Vector` необходимо указать название модели эмбеддингов в поле `VECTOR_MODEL_NAME` в файле `.env`.

Также для использования в режиме `Vector` необходимо стабильное подключение к `HuggingFaceHub`. При невозможности подключения скрипт выдаст предупреждение и будет переключен на `Non-vector` режим.

Для классификации и принятия решения вызовите метод `decide` (необходимо передать список пользовательских запросов):
```python
result = classifier.decide(prompts)
```
или
```python
result = classifier.decide([prompt])
```

Метод возвращает список объектов `DecisionResult` со следующими полями:
- `verdict` - принятое решение/классификация (`"ALLOW"`/`"BLOCK"`)
- `probability` - предсказанная моделью вероятность вредоносности (`0 <= probability <= 1`)
- `score` - рейтинг вредоносности, вычисляемый по формуле `score=-log(1 - probability + 1e-6)`, (`-10 < score < 14`)
- `features` - признаки запроса, на основании которых был получен результат

## Contribution
Suggestions and pull requests are welcomed!