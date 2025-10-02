package com.cor.driveremulationservice.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

@Slf4j
@Component
public class DriverWebSocketHandler extends TextWebSocketHandler {

    private WebSocketSession currentSession;
    private final Random random = new Random();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        if (currentSession != null && currentSession.isOpen()) {
            session.close(CloseStatus.SESSION_NOT_RELIABLE);
            log.warn("Rejected new connection - already have active session");
            return;
        }

        currentSession = session;
        log.info("WebSocket connection established: {}", session.getId());
        session.sendMessage(new TextMessage("DRIVER_READY"));
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        if (session.equals(currentSession)) {
            currentSession = null;
        }
        log.info("WebSocket connection closed: {}", session.getId());
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String payload = message.getPayload();
        log.info("Received command: {}", payload);

        if ("START".equals(payload)) {
            startParallelDataTransmission(session);
        } else if ("STOP".equals(payload)) {
            stopDataTransmission();
        } else {
            session.sendMessage(new TextMessage("ERROR:Unknown command"));
        }
    }

    private void startParallelDataTransmission(WebSocketSession session) {
        new Thread(() -> {
            try {
                log.info("🚀 Starting PARALLEL data transmission with random files");
                session.sendMessage(new TextMessage("TRANSMISSION_STARTED"));

                String randomBpmFile = getRandomCsvFile("bpm");
                String randomUterusFile = getRandomCsvFile("uterus");

                log.info("🎲 Selected random files - BPM: {}, Uterus: {}", randomBpmFile, randomUterusFile);

                List<String> bpmLines = loadCsvFile("bpm", randomBpmFile);
                List<String> uterusLines = loadCsvFile("uterus", randomUterusFile);

                log.info("📊 Loaded BPM lines: {}, Uterus lines: {}", bpmLines.size(), uterusLines.size());

                session.sendMessage(new TextMessage("FILE_INFO:bpm:" + randomBpmFile));
                session.sendMessage(new TextMessage("FILE_INFO:uterus:" + randomUterusFile));

                session.sendMessage(new TextMessage("FILE_START:bpm"));
                session.sendMessage(new TextMessage("FILE_START:uterus"));

                sendParallelData(session, bpmLines, uterusLines);

                session.sendMessage(new TextMessage("FILE_END:bpm"));
                session.sendMessage(new TextMessage("FILE_END:uterus"));

                session.sendMessage(new TextMessage("TRANSMISSION_COMPLETED"));
                log.info("✅ Parallel data transmission completed with random files");

            } catch (Exception e) {
                log.error("💥 Error during parallel data transmission", e);
                try {
                    session.sendMessage(new TextMessage("ERROR:" + e.getMessage()));
                } catch (Exception ex) {
                    log.error("Error sending error message", ex);
                }
            }
        }).start();
    }

    private String getRandomCsvFile(String dataType) throws Exception {
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            java.net.URL resource = classLoader.getResource("data/" + dataType);

            if (resource != null) {
                Path dirPath = Paths.get(resource.toURI());
                List<Path> csvFiles = Files.list(dirPath)
                        .filter(Files::isRegularFile)
                        .filter(p -> p.toString().toLowerCase().endsWith(".csv"))
                        .collect(Collectors.toList());

                if (!csvFiles.isEmpty()) {
                    Path randomFile = csvFiles.get(random.nextInt(csvFiles.size()));
                    log.info("🎲 Found {} CSV files in classpath/{}, selected: {}",
                            csvFiles.size(), dataType, randomFile.getFileName());
                    return randomFile.getFileName().toString();
                }
            }
        } catch (Exception e) {
            log.warn("⚠️ Failed to find files in classpath, trying filesystem...");
        }

        try {
            String[] possiblePaths = {
                    "src/main/resources/data/" + dataType,
                    "data/" + dataType,
                    "/app/data/" + dataType
            };

            for (String path : possiblePaths) {
                Path dirPath = Paths.get(path);
                if (Files.exists(dirPath) && Files.isDirectory(dirPath)) {
                    List<Path> csvFiles = Files.list(dirPath)
                            .filter(Files::isRegularFile)
                            .filter(p -> p.toString().toLowerCase().endsWith(".csv"))
                            .collect(Collectors.toList());

                    if (!csvFiles.isEmpty()) {
                        Path randomFile = csvFiles.get(random.nextInt(csvFiles.size()));
                        log.info("🎲 Found {} CSV files in {}, selected: {}",
                                csvFiles.size(), path, randomFile.getFileName());
                        return randomFile.getFileName().toString();
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to find files in filesystem", e);
        }

        throw new Exception("No CSV files found for: " + dataType);
    }

    private List<String> loadCsvFile(String dataType, String fileName) throws Exception {
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            String resourcePath = "data/" + dataType + "/" + fileName;
            java.net.URL resource = classLoader.getResource(resourcePath);

            if (resource != null) {
                Path filePath = Paths.get(resource.toURI());
                List<String> lines = Files.readAllLines(filePath);
                log.info("✅ Loaded {} data from classpath: {} lines from {}",
                        dataType, lines.size(), fileName);
                return lines;
            }
        } catch (Exception e) {
            log.warn("⚠️ Failed to load from classpath, trying filesystem...");
        }

        try {
            String[] possiblePaths = {
                    "src/main/resources/data/" + dataType + "/" + fileName,
                    "data/" + dataType + "/" + fileName,
                    "/app/data/" + dataType + "/" + fileName
            };

            for (String path : possiblePaths) {
                Path filePath = Paths.get(path);
                if (Files.exists(filePath)) {
                    List<String> lines = Files.readAllLines(filePath);
                    log.info("✅ Loaded {} data from filesystem: {} lines from {}",
                            dataType, lines.size(), filePath.toAbsolutePath());
                    return lines;
                }
            }
        } catch (Exception e) {
            log.error("Failed to load from filesystem", e);
        }

        throw new Exception("Data file not found: " + dataType + "/" + fileName);
    }

    private void sendParallelData(WebSocketSession session, List<String> bpmLines, List<String> uterusLines) throws Exception {
        int bpmIndex = 1;
        int uterusIndex = 1;
        int maxLines = Math.min(bpmLines.size(), uterusLines.size());
        int dataPoints = maxLines - 1;

        log.info("🔄 Sending {} parallel data points", dataPoints);

        int sentPoints = 0;
        while (bpmIndex < maxLines && uterusIndex < maxLines) {
            if (!session.isOpen()) {
                log.warn("Session closed, stopping transmission");
                break;
            }

            String bpmLine = bpmLines.get(bpmIndex);
            String uterusLine = uterusLines.get(uterusIndex);

            session.sendMessage(new TextMessage("DATA:bpm:" + bpmLine));
            session.sendMessage(new TextMessage("DATA:uterus:" + uterusLine));

            sentPoints++;

            if (sentPoints <= 3 || sentPoints % 10 == 0) {
                log.debug("Sent parallel data point {}: bpm={}, uterus={}",
                        sentPoints, bpmLine, uterusLine);
            }

            bpmIndex++;
            uterusIndex++;
            Thread.sleep(200);
        }

        log.info("Successfully sent {} parallel data points", sentPoints);

        if (bpmLines.size() != uterusLines.size()) {
            log.warn("Files have different lengths: BPM={}, Uterus={}",
                    bpmLines.size(), uterusLines.size());
        }
    }

    private void stopDataTransmission() {
        log.info("Data transmission stop requested");
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        log.error("WebSocket transport error", exception);
        if (session.equals(currentSession)) {
            currentSession = null;
        }
    }
}