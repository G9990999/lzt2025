FROM eclipse-temurin:21-jdk-alpine

WORKDIR /app

COPY build/libs/DriverEmulationService-0.0.1-SNAPSHOT.jar app.jar

COPY data/ ./data/

EXPOSE 8080

ENTRYPOINT ["java", "-XX:+UseShenandoahGC", "-jar", "app.jar"]
