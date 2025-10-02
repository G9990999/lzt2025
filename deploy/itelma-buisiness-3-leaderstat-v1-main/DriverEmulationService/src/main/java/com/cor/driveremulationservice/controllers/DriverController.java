package com.cor.driveremulationservice.controllers;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/driver")
public class DriverController {

    @GetMapping("/status")
    public ResponseEntity<Map<String, String>> getStatus() {
        Map<String, String> status = new HashMap<>();
        status.put("status", "READY");
        status.put("message", "Driver emulator is running");
        status.put("websocket_endpoint", "ws://localhost:8080/driver-data");
        status.put("data_files", "bpm/data.csv, uterus/data.csv");
        return ResponseEntity.ok(status);
    }
}
