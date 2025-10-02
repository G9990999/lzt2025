    """
    Теперь этот файл можно прямо скормить CatBoostClassifier, как я показывал в предыдущем коде.
    """

import pandas as pd

# Загружаем датасет
data = pd.read_csv("ctg_dataset.csv")
print(data.head())

# Разделяем на X и y
X = data[["BPM", "Uterus"]]
y = data["Target"]
