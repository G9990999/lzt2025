# Java Spring Boot + ONNX Runtime + PostgreSQL Service

## 📌 Описание
REST API сервис на **Java Spring Boot** для инференса модели в формате **ONNX**.  
Поддерживает 3 класса: **Regular, Hypoxia, Prediction**.  
Все предсказания сохраняются в **PostgreSQL**.  
Встроен **Swagger UI** для тестирования API.

---

## 🚀 Запуск через Docker Compose

```bash
docker-compose build
docker-compose up
```

Сервис запустится на:  
👉 [http://localhost:8080](http://localhost:8080)  
Swagger UI: 👉 [http://localhost:8080/swagger-ui.html](http://localhost:8080/swagger-ui.html)  
База данных: **PostgreSQL (порт 5432)**

---

## 🔹 Примеры API

### 🔸 1. Сделать предсказание
```bash
curl -X POST http://localhost:8080/api/predict      -H "Content-Type: application/json"      -d '{"features":[1.2, 3.4, 5.6, 7.8]}'
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

### 🔸 2. Получить историю предсказаний
```bash
curl http://localhost:8080/api/logs
```

---

## 🛠 Структура проекта

```
springboot-onnx-demo/
 ├─ src/main/java/com/example/onnx/
 │   ├─ OnnxService.java
 │   ├─ PredictionController.java
 │   ├─ Prediction.java
 │   ├─ PredictionRepository.java
 │   └─ SpringbootOnnxDemoApplication.java
 ├─ src/main/resources/
 │   ├─ application.properties
 │   └─ model.onnx
 ├─ Dockerfile
 ├─ docker-compose.yml
 └─ db/
     └─ init.sql
```

---

## 📂 Таблица в PostgreSQL

```sql
CREATE TABLE prediction (
    id SERIAL PRIMARY KEY,
    features TEXT,
    predicted_class VARCHAR(50),
    probabilities TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
