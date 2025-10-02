package com.cor.monitorservice.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.WebSocketMessage;
import org.springframework.web.socket.WebSocketSession;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Component
public class FrontendWebSocketHandler implements WebSocketHandler {

    private final Map<String, WebSocketSession> frontendSessions = new ConcurrentHashMap<>();
    private final DriverConnectionService driverConnectionService;

    public FrontendWebSocketHandler(@Lazy DriverConnectionService driverConnectionService) {
        this.driverConnectionService = driverConnectionService;
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        frontendSessions.put(session.getId(), session);
        log.info("Frontend connected: {}", session.getId());
        session.sendMessage(new TextMessage("MAIN_APP_READY"));
    }

    @Override
    public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) throws Exception {
        String payload = message.getPayload().toString();
        log.info("Received from frontend: {}", payload);

        if (payload.startsWith("START:")) {
            String snils = payload.substring(6);
            handleStartCommand(session, snils);
        } else if ("STATUS".equals(payload)) {
            sendStatus(session);
        } else {
            session.sendMessage(new TextMessage("ERROR:Unknown command"));
        }
    }

    private void handleStartCommand(WebSocketSession session, String snils) {
        if (!isValidSnils(snils)) {
            try {
                session.sendMessage(new TextMessage("ERROR:Invalid SNILS format"));
            } catch (Exception e) {
                log.error("Error sending error message", e);
            }
            return;
        }

        driverConnectionService.startDataTransmission(snils, session.getId());

        try {
            session.sendMessage(new TextMessage("TRANSMISSION_STARTED"));
        } catch (Exception e) {
            log.error("Error sending confirmation", e);
        }
    }

    private boolean isValidSnils(String snils) {
        return snils != null && snils.matches("\\d{3}-\\d{3}-\\d{3} \\d{2}");
    }

    private void sendStatus(WebSocketSession session) throws Exception {
        String status = driverConnectionService.isDriverConnected() ?
                "DRIVER_CONNECTED" : "DRIVER_DISCONNECTED";
        session.sendMessage(new TextMessage("STATUS:" + status));
    }

    public void sendToFrontend(String sessionId, String message) {
        WebSocketSession session = frontendSessions.get(sessionId);
        if (session != null && session.isOpen()) {
            try {
                session.sendMessage(new TextMessage(message));
            } catch (Exception e) {
                log.error("Error sending to frontend", e);
            }
        }
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        log.error("Transport error for session: {}", session.getId(), exception);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus closeStatus) throws Exception {
        frontendSessions.remove(session.getId());
        log.info("Frontend disconnected: {}", session.getId());
    }

    @Override
    public boolean supportsPartialMessages() {
        return false;
    }
}