/**
 * ═══════════════════════════════════════════════════════════════
 * BORDER SENTINEL — Hardware Configuration
 * ═══════════════════════════════════════════════════════════════
 * Pin mapping, radio parameters, and system constants for the
 * ESP32 + ICS-43434 MEMS + SX1276 LoRa edge sensor node.
 */

#ifndef __CONFIG_H__
#define __CONFIG_H__

/* ─── Node Identity ─── */
#define NODE_ID                     "01"
#define FIRMWARE_VERSION            "2.0.0"
#define HARDWARE_REVISION           "B"

/* ─── Audio Configuration ─── */
#define SAMPLE_RATE                 22050
#define AUDIO_FRAME_DURATION_MS     1000
#define AUDIO_BUFFER_SIZE           (SAMPLE_RATE * AUDIO_FRAME_DURATION_MS / 1000)
#define AUDIO_BIT_DEPTH             16
#define AUDIO_CAPTURE_TIMEOUT_MS    2000

/* I2S Pin Mapping — ICS-43434 MEMS Microphone */
#define I2S_PORT_NUM                I2S_NUM_0
#define I2S_BCK_PIN                 26    /* Bit Clock */
#define I2S_WS_PIN                  25    /* Word Select (L/R) */
#define I2S_DATA_IN_PIN             22    /* Serial Data In */
#define I2S_DMA_BUF_COUNT           8
#define I2S_DMA_BUF_LEN             1024

/* Wake-on-Sound */
#define WAKE_ON_SOUND_GPIO          GPIO_NUM_34
#define WAKE_ON_SOUND_THRESHOLD     0.005f

/* ─── MFCC Feature Extraction ─── */
#define N_MFCC                      40
#define N_MELS                      128
#define FFT_SIZE                    2048
#define HOP_LENGTH                  512
#define MFCC_TIME_STEPS             44

/* ─── ML Model ─── */
#define NUM_CLASSES                 3
#define TENSOR_ARENA_SIZE           (80 * 1024)   /* 80 KB */
#define MODEL_INPUT_SIZE            (N_MFCC * MFCC_TIME_STEPS * sizeof(float))

/* Decision Thresholds */
#define FOOTSTEP_INTRUSION_THRESHOLD    0.85f
#define FOOTSTEP_SUSPICIOUS_THRESHOLD   0.70f
#define GUNSHOT_CRITICAL_THRESHOLD      0.70f
#define GUNSHOT_SUSPICIOUS_THRESHOLD    0.50f
#define DETECTION_COOLDOWN_MS           3000

/* ─── LoRa SX1276 Configuration ─── */
#define LORA_FREQUENCY              915000000   /* 915 MHz ISM band */
#define LORA_SPREADING_FACTOR       7
#define LORA_BANDWIDTH              125000      /* 125 kHz */
#define LORA_TX_POWER               14          /* dBm */
#define LORA_CODING_RATE            5           /* 4/5 */
#define LORA_PREAMBLE_LENGTH        8
#define LORA_SYNC_WORD              0x12

/* SPI Pin Mapping — SX1276 LoRa Module */
#define LORA_SPI_HOST               HSPI_HOST
#define LORA_SCK_PIN                18
#define LORA_MISO_PIN               19
#define LORA_MOSI_PIN               23
#define LORA_CS_PIN                 5
#define LORA_RST_PIN                14
#define LORA_DIO0_PIN               2
#define LORA_DIO1_PIN               4

/* ─── Power Management ─── */
#define BATTERY_ADC_CHANNEL         ADC1_CHANNEL_6  /* GPIO 34 */
#define BATTERY_CHECK_INTERVAL_MS   30000           /* 30s */
#define DEEP_SLEEP_DURATION_US      (60 * 1000000)  /* 60s */
#define LOW_BATTERY_THRESHOLD_V     3.2f

/* ─── FreeRTOS Task Priorities ─── */
#define AUDIO_TASK_PRIORITY         (configMAX_PRIORITIES - 1)
#define INFERENCE_TASK_PRIORITY     (configMAX_PRIORITIES - 2)
#define LORA_TASK_PRIORITY          (configMAX_PRIORITIES - 3)

/* ─── Watchdog ─── */
#define WATCHDOG_TIMEOUT_S          30
#define STATUS_LOG_INTERVAL_MS      10000

/* ─── LoRa Packet Structure ─── */
typedef struct __attribute__((packed)) {
    char     node_id[4];
    uint32_t sequence_num;
    char     event_type[16];
    float    confidence;
    uint8_t  alert_level;
    float    battery_voltage;
    float    temperature;
    float    predictions[NUM_CLASSES];
    uint16_t crc;
} lora_packet_t;


#endif /* __CONFIG_H__ */
