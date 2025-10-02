# Fetal Monitor Data Analysis (CatBoost + ONNX)

## 📂 Project structure
```
.
├── datasets/         # Raw data (regular, suspicious, hypoxia)
├── charts/           # Charts (confusion matrix, feature importance, class means)
├── reports/          # Automatically generated reports (CSV, HTML)
├── fetal_monitor_analysis.ipynb  # Jupyter Notebook with code
├── README_ru.md      # Documentation (RU)
├── README_en.md      # Documentation (EN)
├── requirements.txt  # Dependencies
```

## 🚀 Running the project

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Jupyter Notebook:
```bash
jupyter notebook fetal_monitor_analysis.ipynb
```
or
```bash
jupyter lab fetal_monitor_analysis.ipynb
```

3. Execute all notebook cells in order.

## 📊 Results
- Trained model is saved in **ONNX** format (`catboost_model.onnx`).
- Scaler is stored in `scaler.pkl`.
- Metrics → `reports/classification_report_*.csv`.
- Class averages → `reports/class_means_*.csv`.
- HTML report → `reports/report_*.html`.
- Visualizations → `charts/` folder.

## 🧪 Example predictions
Fill `live_data.csv` with new records (BPM, Uterus).  
The model will predict the class (`Regular`, `Suspicious`, `Hypoxia`) and probabilities for each category.
