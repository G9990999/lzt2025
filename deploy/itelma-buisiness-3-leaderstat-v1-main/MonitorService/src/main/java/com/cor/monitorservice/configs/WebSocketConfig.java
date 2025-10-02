package com.cor.monitorservice.configs;

import com.cor.monitorservice.services.DriverConnectionService;
import com.cor.monitorservice.services.FrontendWebSocketHandler;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
@RequiredArgsConstructor
public class WebSocketConfig implements WebSocketConfigurer {

    private final FrontendWebSocketHandler frontendWebSocketHandler;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(frontendWebSocketHandler, "/frontend")
                .setAllowedOriginPatterns("*");
    }

    @Bean
    public boolean connectToDriverOnStartup(DriverConnectionService driverConnectionService) {
        driverConnectionService.connectToDriver();
        return true;
    }
}
