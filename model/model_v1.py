import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# ==== 1. Генерация условных данных ====
# Например, у нас есть BPM (ЧСС плода), Uterus (сила схваток), и метка - состояние плода
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "BPM": np.random.randint(100, 180, size=n),
    "Uterus": np.random.randint(10, 90, size=n),
})

# Целевая переменная (для примера: 0 - норма, 1 - подозрение, 2 - патология)
data["Target"] = np.where(data["BPM"] < 110, 2, 0)
data.loc[(data["BPM"] > 160) | (data["Uterus"] > 70), "Target"] = 1

X = data[["BPM", "Uterus"]]
y = data["Target"]

# ==== 2. Обучение модели ====
train_pool = Pool(X, y)
model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function="MultiClass",
    verbose=False
)

model.fit(train_pool)

# ==== 3. Экспорт модели в ONNX ====
onnx_path = "catboost_ctg.onnx"
model.save_model(
    onnx_path,
    format="onnx",
    export_parameters={
        "onnx_domain": "ai.catboost",
        "onnx_model_version": 1,
        "onnx_doc_string": "CatBoost CTG model",
        "onnx_graph_name": "CatBoostCTG",
    }
)

print(f"Модель успешно сохранена в {onnx_path}")

# ==== 4. Пример предсказания ====
test_sample = np.array([[140, 30]])  # BPM=140, Uterus=30
pred_class = model.predict(test_sample)
print("Прогноз:", pred_class)


import numpy as np
import pandas as pd
import onnxruntime as ort

# ==== Загрузка модели ====
onnx_path = "catboost_ctg.onnx"
session = ort.InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ==== Функции ====
def predict_batch_with_df(samples: np.ndarray) -> pd.DataFrame:
    """
    Предсказание для батча пациентов.
    Возвращает DataFrame с колонками:
    BPM, Uterus, PredictedClass, P0, P1, P2
    """
    pred = session.run([output_name], {input_name: samples.astype(np.float32)})[0]
    classes = np.argmax(pred, axis=1)

    df = pd.DataFrame(samples, columns=["BPM", "Uterus"])
    df["PredictedClass"] = classes
    for i in range(pred.shape[1]):
        df[f"P{i}"] = pred[:, i]
    return df

# ==== Пример ====
batch_samples = np.array([
    [140, 30],  # норма
    [170, 50],  # тахикардия
    [100, 80]   # брадикардия + сильные схватки
])

df_results = predict_batch_with_df(batch_samples)
print(df_results)

import numpy as np
import onnxruntime as ort

# ==== Загрузка модели ====
onnx_path = "catboost_ctg.onnx"
session = ort.InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ==== Функции ====
def predict_patient(bpm: float, uterus: float):
    """
    Предсказание для одного пациента.
    Возвращает (predicted_class, probabilities).
    """
    sample = np.array([[bpm, uterus]], dtype=np.float32)
    pred = session.run([output_name], {input_name: sample})[0]
    predicted_class = int(np.argmax(pred, axis=1)[0])
    probabilities = pred[0].tolist()
    return predicted_class, probabilities

def predict_batch(samples: np.ndarray):
    """
    Предсказание для батча пациентов.
    Возвращает (classes, probabilities).
    classes: массив предсказанных классов.
    probabilities: массив вероятностей для каждого примера.
    """
    pred = session.run([output_name], {input_name: samples.astype(np.float32)})[0]
    classes = np.argmax(pred, axis=1)
    return classes, pred

# ==== Примеры использования ====
cls, probs = predict_patient(140, 30)
print("Один пациент → класс:", cls, "| вероятности:", probs)

batch_samples = np.array([
    [140, 30],  # норма
    [170, 50],  # тахикардия
    [100, 80]   # брадикардия + сильные схватки
])
classes, probabilities = predict_batch(batch_samples)
print("Батч → классы:", classes)
print("Вероятности:\n", probabilities)
