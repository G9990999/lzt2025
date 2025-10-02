package com.cor.monitorservice.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.WebSocketHttpHeaders;
import org.springframework.web.socket.WebSocketMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.client.WebSocketClient;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;

import java.net.URI;
import java.util.concurrent.CompletableFuture;

@Slf4j
@Service
public class DriverConnectionService {

    private WebSocketSession driverSession;
    private String currentSnils;
    @Value("${driver.ws.url}")
    private String url;
    private String currentFrontendSessionId;
    private final FrontendWebSocketHandler frontendHandler;
    private final DataFileService dataFileService;

    public DriverConnectionService(FrontendWebSocketHandler frontendHandler, DataFileService dataFileService) {
        this.frontendHandler = frontendHandler;
        this.dataFileService = dataFileService;
    }

    public void connectToDriver() {
        try {
            String driverUrl = url;
            log.info("[1/5] Starting connection to driver emulator");
            log.info("[2/5] Target URL: {}", driverUrl);

            WebSocketClient client = new StandardWebSocketClient();
            log.info("[3/5] WebSocket client created");

            CompletableFuture<WebSocketSession> futureSession = client.execute(
                    new WebSocketHandler() {
                        @Override
                        public void afterConnectionEstablished(WebSocketSession session) {
                            log.info("[4/5] SUCCESS: WebSocket connection established to driver emulator");
                            log.info("Session ID: {}, URI: {}", session.getId(), session.getUri());
                            driverSession = session;
                        }

                        @Override
                        public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) {
                            log.debug("Received message from driver: {}", message.getPayload());
                            handleDriverMessage(message.getPayload().toString());
                        }

                        @Override
                        public void handleTransportError(WebSocketSession session, Throwable exception) {
                            log.error("[4/5] TRANSPORT ERROR: Driver connection failed", exception);
                            log.error("Error details: {}", exception.getMessage());
                        }

                        @Override
                        public void afterConnectionClosed(WebSocketSession session, CloseStatus closeStatus) {
                            log.info("[5/5] Connection closed: {}", closeStatus);
                            log.info("Close reason: {}, code: {}", closeStatus.getReason(), closeStatus.getCode());
                            driverSession = null;
                        }

                        @Override
                        public boolean supportsPartialMessages() {
                            return false;
                        }
                    },
                    new WebSocketHttpHeaders(),
                    URI.create(driverUrl)
            );

            log.info("[4/5] Connection attempt initiated, waiting for handshake...");

            futureSession.whenComplete((session, throwable) -> {
                if (throwable != null) {
                    log.error("[4/5] COMPLETION ERROR: Failed to complete connection to driver");
                    log.error("Error type: {}", throwable.getClass().getSimpleName());
                    log.error("Error message: {}", throwable.getMessage());

                    if (throwable.getCause() != null) {
                        log.error("Root cause: {}", throwable.getCause().getMessage());
                    }
                } else {
                    log.info("[4/5] FUTURE COMPLETED: Successfully established WebSocket session");
                    if (session != null) {
                        log.info("Session details: ID={}, Open={}", session.getId(), session.isOpen());
                    }
                }
            });

            log.info("[5/5] Connection process completed without exceptions");

        } catch (Exception e) {
            log.error("[X/5] EXCEPTION in connectToDriver method");
            log.error("Exception type: {}", e.getClass().getSimpleName());
            log.error("Exception message: {}", e.getMessage());
            log.error("Stack trace:", e);
        }
    }

    public void startDataTransmission(String snils, String frontendSessionId) {
        log.info("🎬 Starting data transmission for SNILS: {}", snils);
        log.info("🔌 Driver session status: {}", (driverSession != null ? "CONNECTED" : "NULL"));

        if (driverSession != null) {
            log.info("Session details: ID={}, Open={}", driverSession.getId(), driverSession.isOpen());
        } else {
            log.error("CRITICAL: Driver session is NULL, cannot start transmission");
        }

        this.currentSnils = snils;
        this.currentFrontendSessionId = frontendSessionId;

        if (driverSession == null || !driverSession.isOpen()) {
            log.error("Cannot start transmission - driver not connected");
            log.info("Attempting to reconnect...");
            connectToDriver();
            frontendHandler.sendToFrontend(frontendSessionId, "ERROR:Driver not connected");
            return;
        }

        try {
            dataFileService.createDataFiles(snils);
            log.info("Sending START command to driver...");
            driverSession.sendMessage(new TextMessage("START"));
            log.info("Started data transmission for SNILS: {}", snils);

        } catch (Exception e) {
            log.error("Error starting transmission", e);
            frontendHandler.sendToFrontend(frontendSessionId, "ERROR:Failed to start transmission");
        }
    }

    private void handleDriverMessage(String message) {
        log.debug("Received from driver: {}", message);

        if (message.startsWith("DATA:")) {
            String[] parts = message.split(":", 3);
            String dataType = parts[1];
            String csvLine = parts[2];

            dataFileService.saveData(currentSnils, dataType, csvLine);

            String frontendMessage = "DATA:" + dataType + ":" + csvLine;
            frontendHandler.sendToFrontend(currentFrontendSessionId, frontendMessage);

        } else if (message.startsWith("FILE_START:")) {
            String dataType = message.substring(11);
            frontendHandler.sendToFrontend(currentFrontendSessionId, "FILE_START:" + dataType);

        } else if (message.startsWith("FILE_END:")) {
            String dataType = message.substring(9);
            frontendHandler.sendToFrontend(currentFrontendSessionId, "FILE_END:" + dataType);

        } else if (message.startsWith("TRANSMISSION_COMPLETED")) {
            frontendHandler.sendToFrontend(currentFrontendSessionId, "TRANSMISSION_COMPLETED");
            dataFileService.closeFiles(currentSnils);

        } else if (message.startsWith("ERROR:")) {
            frontendHandler.sendToFrontend(currentFrontendSessionId, "DRIVER_ERROR:" + message.substring(6));
        }
    }

    public boolean isDriverConnected() {
        return driverSession != null && driverSession.isOpen();
    }
}