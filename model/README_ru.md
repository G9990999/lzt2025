# Анализ данных фетального монитора (CatBoost + ONNX)

## 📂 Структура проекта
```
.
├── datasets/         # Сырые данные (regular, suspicious, hypoxia)
├── charts/           # Графики (confusion matrix, feature importance, средние значения)
├── reports/          # Автоматически сгенерированные отчёты (CSV, HTML)
├── fetal_monitor_analysis.ipynb  # Jupyter Notebook с кодом
├── README_ru.md      # Документация (RU)
├── README_en.md      # Documentation (EN)
├── requirements.txt  # Список зависимостей
```

## 🚀 Запуск проекта

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите Jupyter Notebook:
```bash
jupyter notebook fetal_monitor_analysis.ipynb
```
или
```bash
jupyter lab fetal_monitor_analysis.ipynb
```

3. Выполните все ячейки ноутбука по порядку.

## 📊 Результаты
- Обученная модель сохраняется в формате **ONNX** (`catboost_model.onnx`).
- Масштабировщик сохраняется в `scaler.pkl`.
- Метрики качества → в `reports/classification_report_*.csv`.
- Средние значения по классам → `reports/class_means_*.csv`.
- HTML-отчёт → `reports/report_*.html`.
- Визуализации → в папке `charts/`.

## 🧪 Пример предсказаний
Файл `live_data.csv` можно заполнить новыми записями (BPM, Uterus).  
Модель предскажет класс (`Regular`, `Suspicious`, `Hypoxia`) и вероятности для каждой категории.
