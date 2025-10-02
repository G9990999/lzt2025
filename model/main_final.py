# %% [markdown]
# # Анализ данных фетального монитора и классификация (CatBoost + ONNX)

# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as ort
import joblib
from IPython.display import Image, display
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from datetime import datetime

# %% [markdown]
# ## 1. Подготовка папок

# %%
os.makedirs("datasets/regular", exist_ok=True)
os.makedirs("datasets/suspicious", exist_ok=True)
os.makedirs("datasets/hypoxia", exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# %% [markdown]
# ## 2. Генерация синтетических данных

# %%
n_regular, n_suspicious, n_hypoxia = 60, 40, 40

regular_data = pd.DataFrame({
    "BPM": np.random.randint(120, 160, size=n_regular),
    "Uterus": np.random.randint(10, 50, size=n_regular),
    "Target": [0] * n_regular
})
regular_data.to_csv("datasets/regular/regular_dataset.csv", index=False)

suspicious_data = pd.DataFrame({
    "BPM": np.random.randint(160, 175, size=n_suspicious),
    "Uterus": np.random.randint(40, 70, size=n_suspicious),
    "Target": [1] * n_suspicious
})
suspicious_data.to_csv("datasets/suspicious/suspicious_dataset.csv", index=False)

hypoxia_data = pd.DataFrame({
    "BPM": np.random.randint(90, 110, size=n_hypoxia),
    "Uterus": np.random.randint(60, 90, size=n_hypoxia),
    "Target": [2] * n_hypoxia
})
hypoxia_data.to_csv("datasets/hypoxia/hypoxia_dataset.csv", index=False)

# %% [markdown]
# ## 3. Объединение и балансировка датасетов

# %%
def build_train_dataset(output_dir=".", balance=True, test_size=0.2):
    files = [
        "datasets/regular/regular_dataset.csv",
        "datasets/suspicious/suspicious_dataset.csv",
        "datasets/hypoxia/hypoxia_dataset.csv"
    ]
    dfs = [pd.read_csv(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)

    if balance:
        min_count = full_df["Target"].value_counts().min()
        full_df = full_df.groupby("Target", group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

    train_df, test_df = train_test_split(full_df, test_size=test_size, stratify=full_df["Target"], random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return train_df, test_df

train_df, test_df = build_train_dataset()

# %% [markdown]
# ## 4. Обучение CatBoost и экспорт в ONNX

# %%
def train_and_export(train_df, test_df, onnx_path="catboost_model.onnx", scaler_path="scaler.pkl"):
    X_train, y_train = train_df[["BPM", "Uterus"]], train_df["Target"]
    X_test, y_test = test_df[["BPM", "Uterus"]], test_df["Target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)

    model = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.1, loss_function="MultiClass", verbose=0)
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test))

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    cm_path = "charts/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    fi = model.get_feature_importance()
    plt.figure(figsize=(5,4))
    plt.bar(["BPM", "Uterus"], fi)
    plt.title("Feature Importance")
    fi_path = "charts/feature_importance.png"
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()

    model.save_model(onnx_path, format="onnx")

    report = classification_report(y_test, y_pred, target_names=["Regular", "Suspicious", "Hypoxia"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_csv = f"reports/classification_report_{timestamp}.csv"
    report_df.to_csv(report_csv)

    class_means = train_df.groupby("Target")["BPM","Uterus"].mean()
    means_csv = f"reports/class_means_{timestamp}.csv"
    class_means.to_csv(means_csv)

    # Барчарты средних значений
    plt.figure()
    class_means["BPM"].plot(kind="bar", color="skyblue", title="Mean BPM by Class")
    bpm_path = "charts/class_means_bpm.png"
    plt.savefig(bpm_path, bbox_inches="tight")
    plt.close()

    plt.figure()
    class_means["Uterus"].plot(kind="bar", color="salmon", title="Mean Uterus by Class")
    uterus_path = "charts/class_means_uterus.png"
    plt.savefig(uterus_path, bbox_inches="tight")
    plt.close()

    return acc, cm_path, fi_path, report_df, class_means, bpm_path, uterus_path, onnx_path, scaler_path

acc, cm_path, fi_path, report_df, class_means, bpm_path, uterus_path, onnx_path, scaler_path = train_and_export(train_df, test_df)

# %% [markdown]
# ## 5. Инференс через onnxruntime + HTML-отчёт

# %%
def predict_with_onnx(onnx_path, scaler_path, samples):
    scaler = joblib.load(scaler_path)
    samples_scaled = scaler.transform(samples)
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: samples_scaled.astype(np.float32)})[0]
    return np.argmax(preds, axis=1), preds

# %%
def predict_live_data(onnx_path, scaler_path, acc, cm_path, fi_path, report_df, class_means, bpm_path, uterus_path,
                      live_file="live_data.csv"):
    class_map = {0: "Regular", 1: "Suspicious", 2: "Hypoxia"}

    if not os.path.exists(live_file):
        pd.DataFrame({"BPM": [145,170,100], "Uterus": [30,55,80]}).to_csv(live_file, index=False)

    live_df = pd.read_csv(live_file)
    class_ids, probs = predict_with_onnx(onnx_path, scaler_path, live_df.values)
    live_df["Predicted"] = [class_map[c] for c in class_ids]
    live_df[["P_Regular","P_Suspicious","P_Hypoxia"]] = probs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/report_{timestamp}.html"

    # HTML отчёт
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Fetal Monitor Report</title></head><body>")
        f.write("<h1>Отчёт по классификации</h1>")
        f.write(f"<p><b>Accuracy:</b> {acc:.3f}</p>")

        f.write("<h2>Classification Report</h2>")
        f.write(report_df.to_html())

        f.write("<h2>Confusion Matrix</h2>")
        f.write(f"<img src='../{cm_path}' width='400'>")

        f.write("<h2>Feature Importance</h2>")
        f.write(f"<img src='../{fi_path}' width='400'>")

        f.write("<h2>Средние значения по классам</h2>")
        f.write(class_means.to_html())
        f.write(f"<img src='../{bpm_path}' width='400'>")
        f.write(f"<img src='../{uterus_path}' width='400'>")

        f.write("<h2>Таблица предсказаний</h2>")
        f.write(live_df.to_html(index=False))

        f.write("</body></html>")

    print(f"HTML-отчёт: {report_path}")
    return live_df

live_results = predict_live_data(onnx_path, scaler_path, acc, cm_path, fi_path, report_df, class_means, bpm_path, uterus_path)
live_results
