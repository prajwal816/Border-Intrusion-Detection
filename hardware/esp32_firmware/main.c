/**
 * ═══════════════════════════════════════════════════════════════
 * BORDER SENTINEL — ESP32 Edge Node Firmware
 * ═══════════════════════════════════════════════════════════════
 * 
 * Target:      ESP32-WROOM-32D (Dual-core Xtensa LX6 @ 240MHz)
 * Peripherals: ICS-43434 MEMS Microphone, SX1276 LoRa Radio
 * Framework:   ESP-IDF v5.1
 * 
 * This firmware captures audio from the MEMS microphone, runs
 * TinyML inference for acoustic classification (footstep / gunshot
 * / noise), and transmits alerts via LoRa to the base station.
 * 
 * Author:  Border Sentinel Team
 * License: MIT
 * ═══════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_sleep.h"
#include "driver/i2s.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "nvs_flash.h"

#include "config.h"
#include "audio_capture.h"
#include "tinyml_inference.h"
#include "lora_driver.h"

static const char *TAG = "SENTINEL_MAIN";

/* ─── Global State ─── */
typedef enum {
    NODE_STATE_BOOT,
    NODE_STATE_IDLE,
    NODE_STATE_LISTENING,
    NODE_STATE_CAPTURING,
    NODE_STATE_PROCESSING,
    NODE_STATE_TRANSMITTING,
    NODE_STATE_DEEP_SLEEP,
    NODE_STATE_ERROR
} node_state_t;

static volatile node_state_t g_node_state = NODE_STATE_BOOT;
static volatile float g_battery_voltage = 3.7f;
static volatile uint32_t g_inference_count = 0;
static volatile uint32_t g_alert_count = 0;
static SemaphoreHandle_t g_state_mutex;

/* Audio ring buffer */
static int16_t g_audio_buffer[AUDIO_BUFFER_SIZE];
static volatile bool g_audio_ready = false;

/* Inference result */
typedef struct {
    float footstep_prob;
    float gunshot_prob;
    float noise_prob;
    char  predicted_class[16];
    float confidence;
    float inference_time_ms;
    uint8_t alert_level;  /* 0=NORMAL, 1=SUSPICIOUS, 2=INTRUSION, 3=HIGH_ALERT */
} inference_result_t;

static inference_result_t g_last_result;


/* ═══════════════════════════════════════════════
 * Wake-on-Sound ISR — Triggered by GPIO interrupt
 * from MEMS microphone comparator output
 * ═══════════════════════════════════════════════ */
static void IRAM_ATTR wake_on_sound_isr(void *arg)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    if (g_node_state == NODE_STATE_LISTENING) {
        g_node_state = NODE_STATE_CAPTURING;
        /* Signal audio capture task */
        vTaskNotifyGiveFromISR((TaskHandle_t)arg, &xHigherPriorityTaskWoken);
    }
    
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}


/* ═══════════════════════════════════════════════
 * Audio Capture Task — I2S DMA transfer from MEMS
 * ═══════════════════════════════════════════════ */
static void audio_capture_task(void *pvParameters)
{
    ESP_LOGI(TAG, "[AUDIO] Capture task started on core %d", xPortGetCoreID());
    
    while (1) {
        /* Wait for wake-on-sound trigger */
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        ESP_LOGI(TAG, "[ESP32 NODE %s] Capturing audio frame...", NODE_ID);
        
        /* Start I2S DMA capture */
        audio_capture_start();
        
        size_t bytes_read = 0;
        esp_err_t err = i2s_read(
            I2S_PORT_NUM,
            g_audio_buffer,
            sizeof(g_audio_buffer),
            &bytes_read,
            pdMS_TO_TICKS(AUDIO_CAPTURE_TIMEOUT_MS)
        );
        
        if (err != ESP_OK || bytes_read < sizeof(g_audio_buffer)) {
            ESP_LOGE(TAG, "[AUDIO] Capture failed: err=%d, bytes=%d", err, bytes_read);
            g_node_state = NODE_STATE_LISTENING;
            continue;
        }
        
        ESP_LOGI(TAG, "[AUDIO] Captured %d samples (%d bytes)", 
                 AUDIO_BUFFER_SIZE, bytes_read);
        
        /* Compute RMS energy for wake-on-sound validation */
        float rms = audio_compute_rms(g_audio_buffer, AUDIO_BUFFER_SIZE);
        ESP_LOGI(TAG, "[AUDIO] RMS energy: %.6f (threshold: %.6f)", 
                 rms, WAKE_ON_SOUND_THRESHOLD);
        
        if (rms < WAKE_ON_SOUND_THRESHOLD) {
            ESP_LOGW(TAG, "[AUDIO] Below threshold — returning to listening");
            g_node_state = NODE_STATE_LISTENING;
            continue;
        }
        
        /* Signal ready for inference */
        g_audio_ready = true;
        g_node_state = NODE_STATE_PROCESSING;
        
        audio_capture_stop();
    }
}


/* ═══════════════════════════════════════════════
 * Inference Task — TinyML model execution
 * ═══════════════════════════════════════════════ */
static void inference_task(void *pvParameters)
{
    ESP_LOGI(TAG, "[ML] Inference task started on core %d", xPortGetCoreID());
    
    /* Initialize TFLite Micro interpreter */
    if (tinyml_init() != ESP_OK) {
        ESP_LOGE(TAG, "[ML] Failed to initialize TinyML engine");
        g_node_state = NODE_STATE_ERROR;
        vTaskDelete(NULL);
        return;
    }
    
    ESP_LOGI(TAG, "[ML] TinyML model loaded — arena size: %d bytes", 
             tinyml_get_arena_size());
    
    while (1) {
        /* Poll for audio data ready */
        if (!g_audio_ready || g_node_state != NODE_STATE_PROCESSING) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        
        ESP_LOGI(TAG, "[ESP32 NODE %s] Running TinyML inference...", NODE_ID);
        
        /* Extract MFCC features from raw audio */
        float mfcc_features[N_MFCC][MFCC_TIME_STEPS];
        int64_t t_start = esp_timer_get_time();
        
        audio_extract_mfcc(g_audio_buffer, AUDIO_BUFFER_SIZE, 
                          (float*)mfcc_features, N_MFCC, MFCC_TIME_STEPS);
        
        /* Run TFLite Micro inference */
        float output[NUM_CLASSES] = {0};
        tinyml_invoke((float*)mfcc_features, output, NUM_CLASSES);
        
        int64_t t_end = esp_timer_get_time();
        float inference_ms = (t_end - t_start) / 1000.0f;
        
        /* Parse results */
        xSemaphoreTake(g_state_mutex, portMAX_DELAY);
        
        g_last_result.footstep_prob = output[0];
        g_last_result.gunshot_prob  = output[1];
        g_last_result.noise_prob    = output[2];
        g_last_result.inference_time_ms = inference_ms;
        
        /* Find predicted class */
        int max_idx = 0;
        float max_prob = output[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (output[i] > max_prob) {
                max_prob = output[i];
                max_idx = i;
            }
        }
        
        const char *class_names[] = {"FOOTSTEP", "GUNSHOT", "NOISE"};
        strncpy(g_last_result.predicted_class, class_names[max_idx], 
                sizeof(g_last_result.predicted_class));
        g_last_result.confidence = max_prob;
        
        /* Apply decision thresholds */
        g_last_result.alert_level = 0;  /* NORMAL */
        
        if (max_idx == 1 && max_prob > GUNSHOT_CRITICAL_THRESHOLD) {
            g_last_result.alert_level = 3;  /* HIGH_ALERT */
        } else if (max_idx == 1 && max_prob > GUNSHOT_SUSPICIOUS_THRESHOLD) {
            g_last_result.alert_level = 2;  /* INTRUSION */
        } else if (max_idx == 0 && max_prob > FOOTSTEP_INTRUSION_THRESHOLD) {
            g_last_result.alert_level = 2;  /* INTRUSION */
        } else if (max_idx == 0 && max_prob > FOOTSTEP_SUSPICIOUS_THRESHOLD) {
            g_last_result.alert_level = 1;  /* SUSPICIOUS */
        }
        
        g_inference_count++;
        if (g_last_result.alert_level >= 2) {
            g_alert_count++;
        }
        
        xSemaphoreGive(g_state_mutex);
        
        ESP_LOGI(TAG, "[ML] Result: %s (%.1f%%) — Alert: %d — Time: %.1fms",
                 g_last_result.predicted_class,
                 g_last_result.confidence * 100.0f,
                 g_last_result.alert_level,
                 inference_ms);
        
        /* Transition to transmit */
        g_audio_ready = false;
        g_node_state = NODE_STATE_TRANSMITTING;
    }
}


/* ═══════════════════════════════════════════════
 * LoRa Transmission Task
 * ═══════════════════════════════════════════════ */
static void lora_tx_task(void *pvParameters)
{
    ESP_LOGI(TAG, "[LoRa] TX task started");
    
    if (lora_init() != ESP_OK) {
        ESP_LOGE(TAG, "[LoRa] Failed to initialize SX1276");
        g_node_state = NODE_STATE_ERROR;
        vTaskDelete(NULL);
        return;
    }
    
    lora_set_spreading_factor(LORA_SPREADING_FACTOR);
    lora_set_bandwidth(LORA_BANDWIDTH);
    lora_set_tx_power(LORA_TX_POWER);
    lora_set_frequency(LORA_FREQUENCY);
    
    ESP_LOGI(TAG, "[LoRa] Radio configured — SF%d, BW%dkHz, %ddBm, %.1fMHz",
             LORA_SPREADING_FACTOR, LORA_BANDWIDTH / 1000,
             LORA_TX_POWER, LORA_FREQUENCY / 1e6);
    
    while (1) {
        if (g_node_state != NODE_STATE_TRANSMITTING) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        
        /* Build LoRa packet */
        lora_packet_t packet;
        memset(&packet, 0, sizeof(packet));
        
        xSemaphoreTake(g_state_mutex, portMAX_DELAY);
        
        strncpy(packet.node_id, NODE_ID, sizeof(packet.node_id));
        packet.sequence_num = g_inference_count;
        strncpy(packet.event_type, g_last_result.predicted_class, 
                sizeof(packet.event_type));
        packet.confidence = g_last_result.confidence;
        packet.alert_level = g_last_result.alert_level;
        packet.battery_voltage = g_battery_voltage;
        packet.temperature = read_internal_temperature();
        packet.predictions[0] = g_last_result.footstep_prob;
        packet.predictions[1] = g_last_result.gunshot_prob;
        packet.predictions[2] = g_last_result.noise_prob;
        
        /* CRC-16 checksum */
        packet.crc = lora_compute_crc16((uint8_t*)&packet, 
                                         sizeof(packet) - sizeof(packet.crc));
        
        xSemaphoreGive(g_state_mutex);
        
        ESP_LOGI(TAG, "[LoRa TX] Sending packet #%lu...", packet.sequence_num);
        ESP_LOGI(TAG, "[LoRa TX] Payload: NODE_ID=%s | EVENT=%s | CONF=%.2f",
                 packet.node_id, packet.event_type, packet.confidence);
        
        /* Transmit */
        esp_err_t tx_err = lora_send_packet((uint8_t*)&packet, sizeof(packet));
        
        if (tx_err == ESP_OK) {
            ESP_LOGI(TAG, "[LoRa TX] Packet sent successfully (%d bytes)", 
                     sizeof(packet));
        } else {
            ESP_LOGE(TAG, "[LoRa TX] Transmission failed: %d", tx_err);
        }
        
        /* Return to listening mode */
        g_node_state = NODE_STATE_LISTENING;
        
        /* Brief cooldown */
        vTaskDelay(pdMS_TO_TICKS(DETECTION_COOLDOWN_MS));
    }
}


/* ═══════════════════════════════════════════════
 * Battery Monitor Task
 * ═══════════════════════════════════════════════ */
static void battery_monitor_task(void *pvParameters)
{
    while (1) {
        /* Read ADC for battery voltage divider */
        uint32_t adc_reading = 0;
        for (int i = 0; i < 16; i++) {
            adc_reading += adc1_get_raw(BATTERY_ADC_CHANNEL);
        }
        adc_reading /= 16;
        
        /* Convert to voltage (3.3V reference, 2:1 divider) */
        g_battery_voltage = (adc_reading / 4095.0f) * 3.3f * 2.0f;
        
        float battery_pct = ((g_battery_voltage - 3.0f) / (4.2f - 3.0f)) * 100.0f;
        battery_pct = fmaxf(0.0f, fminf(100.0f, battery_pct));
        
        ESP_LOGI(TAG, "[BAT] Voltage: %.2fV (%.0f%%)", g_battery_voltage, battery_pct);
        
        /* Low battery warning */
        if (battery_pct < 10.0f) {
            ESP_LOGW(TAG, "[BAT] ⚠ LOW BATTERY — entering deep sleep");
            g_node_state = NODE_STATE_DEEP_SLEEP;
            esp_deep_sleep(DEEP_SLEEP_DURATION_US);
        }
        
        vTaskDelay(pdMS_TO_TICKS(BATTERY_CHECK_INTERVAL_MS));
    }
}


/* ═══════════════════════════════════════════════
 * Main Entry Point
 * ═══════════════════════════════════════════════ */
void app_main(void)
{
    ESP_LOGI(TAG, "═══════════════════════════════════════════");
    ESP_LOGI(TAG, "  BORDER SENTINEL — ESP32 Edge Node v2.0");
    ESP_LOGI(TAG, "  Node ID: %s", NODE_ID);
    ESP_LOGI(TAG, "  CPU: %d MHz | Flash: %d MB", 
             CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ, 4);
    ESP_LOGI(TAG, "═══════════════════════════════════════════");
    
    /* Initialize NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    
    /* Create mutex */
    g_state_mutex = xSemaphoreCreateMutex();
    
    /* Initialize peripherals */
    ESP_LOGI(TAG, "[INIT] Configuring I2S for MEMS microphone...");
    audio_i2s_init();
    
    ESP_LOGI(TAG, "[INIT] Configuring SPI for LoRa SX1276...");
    lora_spi_init();
    
    ESP_LOGI(TAG, "[INIT] Configuring wake-on-sound GPIO...");
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_POSEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << WAKE_ON_SOUND_GPIO),
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
    };
    gpio_config(&io_conf);
    
    /* Create FreeRTOS tasks */
    TaskHandle_t audio_task_handle;
    
    xTaskCreatePinnedToCore(audio_capture_task, "audio_cap", 8192, NULL,
                            AUDIO_TASK_PRIORITY, &audio_task_handle, 0);
    
    xTaskCreatePinnedToCore(inference_task, "inference", 16384, NULL,
                            INFERENCE_TASK_PRIORITY, NULL, 1);
    
    xTaskCreate(lora_tx_task, "lora_tx", 4096, NULL,
                LORA_TASK_PRIORITY, NULL);
    
    xTaskCreate(battery_monitor_task, "battery", 2048, NULL,
                tskIDLE_PRIORITY + 1, NULL);
    
    /* Attach wake-on-sound ISR */
    gpio_install_isr_service(0);
    gpio_isr_handler_add(WAKE_ON_SOUND_GPIO, wake_on_sound_isr, 
                         (void*)audio_task_handle);
    
    /* System ready */
    g_node_state = NODE_STATE_LISTENING;
    ESP_LOGI(TAG, "[INIT] System ready — entering listening mode");
    ESP_LOGI(TAG, "[ESP32 NODE %s] Wake-on-sound active", NODE_ID);
    
    /* Main loop — watchdog feed + status logging */
    while (1) {
        ESP_LOGI(TAG, "[STATUS] State: %d | Inferences: %lu | Alerts: %lu | Battery: %.2fV",
                 g_node_state, g_inference_count, g_alert_count, g_battery_voltage);
        vTaskDelay(pdMS_TO_TICKS(STATUS_LOG_INTERVAL_MS));
    }
}
