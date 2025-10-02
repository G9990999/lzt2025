# Go + ONNX Runtime + PostgreSQL Service

## 📌 Описание
REST API сервис на **Go** для инференса модели в формате **ONNX**.  
Поддерживает 3 класса: **Regular, Hypoxia, Prediction**.  
Все предсказания логируются в **PostgreSQL**.

---

## 🚀 Запуск через Docker Compose

```bash
docker-compose build
docker-compose up
```

Сервис запустится на:  
👉 [http://localhost:8080](http://localhost:8080)  
База данных: **PostgreSQL (порт 5432)**

---

## 🔹 Примеры API

### 🔸 1. Сделать предсказание
```bash
curl -X POST http://localhost:8080/predict      -H "Content-Type: application/json"      -d '{"features":[1.2, 3.4, 5.6, 7.8]}'
```

Пример ответа:
```json
{
  "predicted_class": "Hypoxia",
  "probabilities": {
    "Regular": 0.12,
    "Hypoxia": 0.76,
    "Prediction": 0.12
  }
}
```

---

## 🛠 Структура проекта

```
go-onnx-demo/
 ├─ main.go
 ├─ model.onnx
 ├─ Dockerfile
 ├─ docker-compose.yml
 └─ db/
     └─ init.sql
```

---

## 📂 Таблица в PostgreSQL

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    features JSONB NOT NULL,
    predicted_class VARCHAR(50) NOT NULL,
    probabilities JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
