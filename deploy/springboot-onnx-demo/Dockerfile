FROM maven:3.9.5-eclipse-temurin-17 AS builder

WORKDIR /app
COPY . .
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jdk-jammy
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
COPY src/main/resources/model.onnx model.onnx

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
