package com.cor.monitorservice.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
public class DataFileService {

    private final Map<String, Map<String, BufferedWriter>> fileWriters = new ConcurrentHashMap<>();
    private final String dataPath;

    public DataFileService() {
        this.dataPath = determineDataPath();
        log.info("📁 Monitor data path: {}", dataPath);
    }

    private String determineDataPath() {
        // Универсальный путь, работает везде
        return "data/";
    }

    public void createDataFiles(String snils) {
        try {
            // Создаем папки в универсальном пути
            Files.createDirectories(Paths.get(dataPath + "bpm"));
            Files.createDirectories(Paths.get(dataPath + "uterus"));

            String sanitizedSnils = snils.replaceAll("[^\\d]", "");
            String fileName = sanitizedSnils + ".csv";

            Map<String, BufferedWriter> writers = new HashMap<>();

            // Создаем файлы в универсальном пути
            BufferedWriter bpmWriter = new BufferedWriter(new FileWriter(dataPath + "bpm/" + fileName));
            writers.put("bpm", bpmWriter);

            BufferedWriter uterusWriter = new BufferedWriter(new FileWriter(dataPath + "uterus/" + fileName));
            writers.put("uterus", uterusWriter);

            fileWriters.put(snils, writers);

            log.info("✅ Created data files for SNILS: {}", snils);
            log.info("📁 BPM file: {}bpm/{}", dataPath, fileName);
            log.info("📁 Uterus file: {}uterus/{}", dataPath, fileName);

        } catch (Exception e) {
            log.error("❌ Error creating data files for SNILS: {}", snils, e);
        }
    }

    public void saveData(String snils, String dataType, String csvLine) {
        Map<String, BufferedWriter> writers = fileWriters.get(snils);
        if (writers != null) {
            BufferedWriter writer = writers.get(dataType);
            if (writer != null) {
                try {
                    writer.write(csvLine);
                    writer.newLine();
                    writer.flush();
                    log.debug("💾 Saved data to {}/{}", dataType, snils);
                } catch (Exception e) {
                    log.error("Error writing data to file for SNILS: {}, type: {}", snils, dataType, e);
                }
            }
        }
    }

    public void closeFiles(String snils) {
        Map<String, BufferedWriter> writers = fileWriters.remove(snils);
        if (writers != null) {
            for (Map.Entry<String, BufferedWriter> entry : writers.entrySet()) {
                try {
                    entry.getValue().close();
                    log.info("🔒 Closed {} file for SNILS: {}", entry.getKey(), snils);
                } catch (Exception e) {
                    log.error("Error closing file for SNILS: {}, type: {}", snils, entry.getKey(), e);
                }
            }
        }
    }
}
