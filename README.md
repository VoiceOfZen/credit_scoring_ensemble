# Репозиторий проекта "Credit Scoring with Ensemble Models"

## Структура проекта

```
credit_score/                 # Корневая папка проекта
├── data/                     # Данные
│   └── default_of_credit_card_clients.xls   # Excel-файл с исходным датасетом
├── notebooks/                # (опционально) Jupyter-ноутбуки
│   └── exploratory_analysis.ipynb
├── src/                      # Исходный код
│   ├── data_utils.py         # Загрузка и предобработка данных
│   ├── models.py             # Определение моделей
│   ├── evaluation.py         # Оценка моделей и визуализации
│   └── credit.py             # Главный скрипт обучения и сохранения результатов
├── results/                  # Сохранённые графики и таблицы
│   ├── roc_logistic_regression.png
│   ├── roc_random_forest.png
│   └── feat_imp_gradient_boosting.png
├── .gitignore                # Список игнорируемых файлов Git
├── requirements.txt          # Список зависимостей проекта
└── README.md                 # Описание проекта и инструкция по запуску
```

## Описание файлов

* **data\_utils.py**: функция `load_and_prepare_data()` для автоматического скачивания и подготовки данных.
* **models.py**: функция `get_models()` возвращает словарь моделей для обучения.
* **evaluation.py**: функции `evaluate_models()` и `save_roc_curve()`, `save_feature_importance()` для оценки и визуализации.
* **credit.py**: основной скрипт, который выполняет загрузку данных, обучение моделей, сохранение метрик и графиков.
* **requirements.txt**: содержит все Python-библиотеки и версии, необходимые для запуска проекта.
* **.gitignore**: исключает из репозитория большая бинарные файлы, виртуальное окружение и папку `__pycache__`.
* **results/**: папка с автоматически сохранёнными изображениями ROC-кривых и графиками важности признаков.
* **data/**: папка с исходным Excel-файлом или скрипт автоматической загрузки.
* **notebooks/**: (опционально) exploratory data analysis и визуализации.

## Инструкция по запуску

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/your_username/credit_score.git
   cd credit_score
   ```
2. Создайте виртуальное окружение и активируйте его:

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Mac/Linux
   .\.venv\Scripts\activate    # Windows
   ```
3. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```
4. Запустите главный скрипт:

   ```bash
   python src/credit.py
   ```
5. Результаты и графики будут сохранены в папке `results/`.

## Дополнительно

* Если вы хотите исследовать данные вручную, воспользуйтесь ноутбуком `notebooks/exploratory_analysis.ipynb`.
* Для повторного скачивания данных запустите `src/data_utils.py` или скрипт в ноутбуке.

