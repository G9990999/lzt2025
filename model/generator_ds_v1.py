import os
import pandas as pd
import numpy as np

# ==== Создаём папки ====
os.makedirs("datasets/regular", exist_ok=True)
os.makedirs("datasets/suspicious", exist_ok=True)
os.makedirs("datasets/hypoxia", exist_ok=True)

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

# ==== Генерация "подозрение" (Target = 1) ====
suspicious_data = pd.DataFrame({
    "BPM": np.random.randint(160, 175, size=n_suspicious),  # высокая ЧСС
    "Uterus": np.random.randint(40, 70, size=n_suspicious), # средняя активность
    "Target": [1] * n_suspicious
})

# ==== Генерация "гипоксия" (Target = 2) ====
hypoxia_data = pd.DataFrame({
    "BPM": np.random.randint(90, 110, size=n_hypoxia),   # низкая ЧСС
    "Uterus": np.random.randint(60, 90, size=n_hypoxia), # высокая активность
    "Target": [2] * n_hypoxia
})

# ==== Пути ====
regular_path = "datasets/regular/regular_dataset.csv"
suspicious_path = "datasets/suspicious/suspicious_dataset.csv"
hypoxia_path = "datasets/hypoxia/hypoxia_dataset.csv"

# ==== Сохраняем ====
regular_data.to_csv(regular_path, index=False, encoding="utf-8")
suspicious_data.to_csv(suspicious_path, index=False, encoding="utf-8")
hypoxia_data.to_csv(hypoxia_path, index=False, encoding="utf-8")

print(f"✅ Создано: {regular_path} ({len(regular_data)} строк)")
print(f"✅ Создано: {suspicious_path} ({len(suspicious_data)} строк)")
print(f"✅ Создано: {hypoxia_path} ({len(hypoxia_data)} строк)")

# ==== Объединяем для обучения ====
all_data = pd.concat([regular_data, suspicious_data, hypoxia_data], ignore_index=True)
print("\nПример объединённого датасета:")
print(all_data.sample(5))
