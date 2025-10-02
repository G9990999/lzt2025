import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as ort
import joblib

# ==== Создание папок ====
os.makedirs("datasets/regular", exist_ok=True)
os.makedirs("datasets/suspicious", exist_ok=True)
os.makedirs("datasets/hypoxia", exist_ok=True)
os.makedirs("charts", exist_ok=True)  # для графиков

# ==== Параметры ====
n_regular = 60
n_suspicious = 40
n_hypoxia = 40

# ==== Генерация "норма" (Target = 0) ====
regular_data = pd.DataFrame({
    "BPM": np.random.randint(120, 160, size=n_regular),
    "Uterus": np.random.randint(10, 50, size=n_regular),
    "Target": [0] * n_regular
})
regular_data.to_csv("datasets/regular/regular_dataset.csv", index=False, encoding="utf-8")

# ==== Генерация "подозрение" (Target = 1) ====
suspicious_data = pd.DataFrame({
    "BPM": np.random.randint(160, 175, size=n_suspicious),
    "Uterus": np.random.randint(40, 70, size=n_suspicious),
    "Target": [1] * n_suspicious
})
suspicious_data.to_csv("datasets/suspicious/suspicious_dataset.csv", index=False, encoding="utf-8")

# ==== Генерация "гипоксия" (Target = 2) ====
hypoxia_data = pd.DataFrame({
    "BPM": np.random.randint(90, 110, size=n_hypoxia),
    "Uterus": np.random.randint(60, 90, size=n_hypoxia),
    "Target": [2] * n_hypoxia
})
hypoxia_data.to_csv("datasets/hypoxia/hypoxia_dataset.csv", index=False, encoding="utf-8")


# ==== Функция объединения, балансировки и разделения ====
def build_train_dataset(output_dir=".", balance=True, test_size=0.2):
    files = [
        "datasets/regular/regular_dataset.csv",
        "datasets/suspicious/suspicious_dataset.csv",
        "datasets/hypoxia/hypoxia_dataset.csv"
    ]
    dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
    full_df = pd.concat(dfs, ignore_index=True)

    if balance:
        counts = full_df["Target"].value_counts()
        min_count = counts.min()
        full_df = (
            full_df.groupby("Target", group_keys=False)
            .apply(lambda x: x.sample(min_count, replace=False, random_state=42))
            .reset_index(drop=True)
        )
        print(f"⚖️ Балансировка: по {min_count} примеров на класс")

    # Сохраняем объединённый датасет
    train_dataset_path = os.path.join(output_dir, "train_dataset.csv")
    full_df.to_csv(train_dataset_path, index=False, encoding="utf-8")
    print(f"📂 Объединённый датасет сохранён: {train_dataset_path} ({len(full_df)} строк)")

    # Делим на train/test
    train_df, test_df = train_test_split(
        full_df, test_size=test_size, random_state=42, stratify=full_df["Target"]
    )
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8")

    print(f"✅ Train: {len(train_df)} строк")
    print(f"✅ Test:  {len(test_df)} строк")

    return train_df, test_df


# ==== Обучение, проверка и экспорт ====
def train_and_export(train_df, test_df, onnx_path="catboost_model.onnx", scaler_path="scaler.pkl"):
    X_train, y_train = train_df[["BPM", "Uterus"]], train_df["Target"]
    X_test, y_test = test_df[["BPM", "Uterus"]], test_df["Target"]

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем scaler
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler сохранён: {scaler_path}")

    # Обучаем CatBoost
    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=0,
        random_seed=42
    )
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test))

    # Предсказания
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n🎯 Accuracy: {acc:.3f}")
    print("📊 Confusion Matrix:")
    print(cm)

    # Визуализация confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Regular (0)", "Suspicious (1)", "Hypoxia (2)"],
                yticklabels=["Regular (0)", "Suspicious (1)", "Hypoxia (2)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Экспорт в ONNX
    model.save_model(onnx_path, format="onnx")
    print(f"💾 Модель сохранена в ONNX: {onnx_path}")

    return onnx_path, scaler_path


# ==== Предсказания через onnxruntime (с вероятностями) ====
def predict_with_onnx(onnx_path, scaler_path, samples):
    scaler = joblib.load(scaler_path)
    samples_scaled = scaler.transform(samples)

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: samples_scaled.astype(np.float32)})[0]

    class_ids = np.argmax(preds, axis=1)       # метки классов
    probabilities = preds                      # вероятности
    return class_ids, probabilities


# ==== Загрузка live данных, предсказания и визуализация ====
def predict_live_data(onnx_path, scaler_path, live_file="live_data.csv", output_file="live_predictions.csv"):
    class_map = {0: "Regular", 1: "Suspicious", 2: "Hypoxia"}

    if not os.path.exists(live_file):
        # создаём пример live_data.csv
        live_df = pd.DataFrame({
            "BPM": [145, 170, 100],
            "Uterus": [30, 55, 80]
        })
        live_df.to_csv(live_file, index=False, encoding="utf-8")
        print(f"📂 Создан пример {live_file}")

    live_df = pd.read_csv(live_file)
    class_ids, probs = predict_with_onnx(onnx_path, scaler_path, live_df.values)

    # добавляем текстовые метки
    live_df["Predicted"] = [class_map[p] for p in class_ids]

    # добавляем вероятности
    live_df["P_Regular"] = probs[:, 0]
    live_df["P_Suspicious"] = probs[:, 1]
    live_df["P_Hypoxia"] = probs[:, 2]

    # сохраняем в CSV
    live_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n🔮 Предсказания для {live_file}:")
    print(live_df)
    print(f"💾 Сохранено в {output_file}")

    # Индивидуальные графики
    for i, row in live_df.iterrows():
        plt.figure(figsize=(5, 4))
        plt.bar(["Regular", "Suspicious", "Hypoxia"],
                [row["P_Regular"], row["P_Suspicious"], row["P_Hypoxia"]],
                color=["green", "orange", "red"])
        plt.title(f"Sample {i+1}: BPM={row['BPM']}, Uterus={row['Uterus']} → {row['Predicted']}")
        plt.ylabel("Probability")
        plt.ylim(0, 1)

        chart_path = f"charts/sample_{i+1}.png"
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        print(f"📊 Сохранён график: {chart_path}")

    # Сводный stacked bar chart
    plt.figure(figsize=(8, 6))
    ind = np.arange(len(live_df))
    plt.bar(ind, live_df["P_Regular"], label="Regular", color="green")
    plt.bar(ind, live_df["P_Suspicious"], bottom=live_df["P_Regular"], label="Suspicious", color="orange")
    plt.bar(ind, live_df["P_Hypoxia"],
            bottom=live_df["P_Regular"] + live_df["P_Suspicious"],
            label="Hypoxia", color="red")

    plt.xticks(ind, [f"S{i+1}" for i in range(len(live_df))])
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.title("Summary Probabilities for Live Data")
    plt.legend()

    summary_path = "charts/summary.png"
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Сохранён общий график: {summary_path}")

    return live_df


# ==== Запуск ====
train_df, test_df = build_train_dataset(balance=True, test_size=0.2)
onnx_path, scaler_path = train_and_export(train_df, test_df,
                                          onnx_path="catboost_model.onnx",
                                          scaler_path="scaler.pkl")

# Предсказания для live_data.csv с графиками в PNG
predict_live_data(onnx_path, scaler_path, live_file="live_data.csv", output_file="live_predictions.csv")
