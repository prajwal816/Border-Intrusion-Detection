/**
 * LoRa Base Station Gateway Firmware
 * ═══════════════════════════════════
 * Receives packets from edge sensor nodes and forwards
 * them to the monitoring dashboard via UART/WiFi.
 * Target: ESP32 Gateway Module
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_http_server.h"
#include "driver/spi_master.h"
#include "driver/uart.h"

#include "../esp32_firmware/config.h"

static const char *TAG = "GATEWAY";

#define GATEWAY_ID          "BASE-01"
#define MAX_REGISTERED_NODES 16
#define WIFI_SSID           "SENTINEL_NET"
#define WIFI_PASS           "s3nt1n3l_2024"
#define HTTP_PORT           8080

typedef struct {
    char     node_id[4];
    uint32_t last_seen;
    float    last_rssi;
    float    battery_pct;
    uint32_t packets_received;
    uint8_t  last_alert_level;
} node_registry_entry_t;

static node_registry_entry_t g_node_registry[MAX_REGISTERED_NODES];
static int g_registered_count = 0;
static uint32_t g_total_packets = 0;
static uint32_t g_total_alerts = 0;
static uint32_t g_crc_failures = 0;


static node_registry_entry_t* _find_or_register_node(const char *node_id)
{
    for (int i = 0; i < g_registered_count; i++) {
        if (strcmp(g_node_registry[i].node_id, node_id) == 0)
            return &g_node_registry[i];
    }
    if (g_registered_count < MAX_REGISTERED_NODES) {
        node_registry_entry_t *entry = &g_node_registry[g_registered_count++];
        strncpy(entry->node_id, node_id, sizeof(entry->node_id));
        entry->packets_received = 0;
        ESP_LOGI(TAG, "New node registered: %s (total: %d)", node_id, g_registered_count);
        return entry;
    }
    return NULL;
}


static void lora_rx_task(void *pvParameters)
{
    ESP_LOGI(TAG, "[BASE STATION] LoRa RX task started");
    ESP_LOGI(TAG, "[BASE STATION] Listening on %.1f MHz, SF7-12 adaptive",
             LORA_FREQUENCY / 1e6);

    /* Initialize LoRa in RX continuous mode */
    lora_init();
    lora_set_frequency(LORA_FREQUENCY);
    lora_set_spreading_factor(LORA_SPREADING_FACTOR);
    lora_set_bandwidth(LORA_BANDWIDTH);

    uint8_t rx_buffer[256];

    while (1) {
        int len = lora_receive_packet(rx_buffer, sizeof(rx_buffer));
        if (len <= 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        g_total_packets++;
        lora_packet_t *pkt = (lora_packet_t *)rx_buffer;

        /* CRC validation */
        uint16_t calc_crc = lora_compute_crc16(rx_buffer, len - sizeof(uint16_t));
        if (calc_crc != pkt->crc) {
            g_crc_failures++;
            ESP_LOGW(TAG, "Packet from NODE-%s: CRC FAILED (dropped)", pkt->node_id);
            continue;
        }

        float rssi = lora_get_packet_rssi();
        float snr  = lora_get_packet_snr();

        ESP_LOGI(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        ESP_LOGI(TAG, "Packet #%lu received from NODE-%s", pkt->sequence_num, pkt->node_id);
        ESP_LOGI(TAG, "  Event: %s | Confidence: %.1f%%",
                 pkt->event_type, pkt->confidence * 100.0f);
        ESP_LOGI(TAG, "  RSSI: %.0f dBm | SNR: %.1f dB", rssi, snr);
        ESP_LOGI(TAG, "  Battery: %.2fV | Temp: %.1f°C",
                 pkt->battery_voltage, pkt->temperature);

        /* Update node registry */
        node_registry_entry_t *entry = _find_or_register_node(pkt->node_id);
        if (entry) {
            entry->last_seen = xTaskGetTickCount();
            entry->last_rssi = rssi;
            entry->battery_pct = ((pkt->battery_voltage - 3.0f) / 1.2f) * 100.0f;
            entry->packets_received++;
            entry->last_alert_level = pkt->alert_level;
        }

        /* Alert processing */
        if (pkt->alert_level >= 2) {
            g_total_alerts++;
            ESP_LOGW(TAG, "🚨 ALERT RECEIVED:");
            ESP_LOGW(TAG, "   Node: %s", pkt->node_id);
            ESP_LOGW(TAG, "   Event: %s", pkt->event_type);
            ESP_LOGW(TAG, "   Confidence: %.0f%%", pkt->confidence * 100);
            ESP_LOGW(TAG, "   Level: %d", pkt->alert_level);

            /* Forward alert via UART to monitoring PC */
            char uart_msg[128];
            snprintf(uart_msg, sizeof(uart_msg),
                     "ALERT|%s|%s|%.2f|%d\n",
                     pkt->node_id, pkt->event_type,
                     pkt->confidence, pkt->alert_level);
            uart_write_bytes(UART_NUM_0, uart_msg, strlen(uart_msg));
        }

        /* Forward full packet via WiFi HTTP API */
        _forward_to_dashboard(pkt, rssi, snr);
    }
}


static void _forward_to_dashboard(lora_packet_t *pkt, float rssi, float snr)
{
    /* In production this POSTs JSON to the Streamlit dashboard */
    ESP_LOGI(TAG, "[HTTP] Forwarding to dashboard: %s -> %s (%.0f%%)",
             pkt->node_id, pkt->event_type, pkt->confidence * 100);
}


static esp_err_t status_handler(httpd_req_t *req)
{
    char resp[512];
    snprintf(resp, sizeof(resp),
        "{\"gateway\":\"%s\",\"nodes\":%d,\"packets\":%lu,"
        "\"alerts\":%lu,\"crc_failures\":%lu}",
        GATEWAY_ID, g_registered_count,
        g_total_packets, g_total_alerts, g_crc_failures);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, strlen(resp));
}


void app_main(void)
{
    ESP_LOGI(TAG, "═══════════════════════════════════════════");
    ESP_LOGI(TAG, "  BORDER SENTINEL — Base Station Gateway");
    ESP_LOGI(TAG, "  Gateway ID: %s", GATEWAY_ID);
    ESP_LOGI(TAG, "═══════════════════════════════════════════");

    /* Start LoRa RX task */
    xTaskCreatePinnedToCore(lora_rx_task, "lora_rx", 8192, NULL, 5, NULL, 0);

    ESP_LOGI(TAG, "[INIT] Gateway online — listening for sensor nodes");

    while (1) {
        ESP_LOGI(TAG, "[STATUS] Nodes: %d | Packets: %lu | Alerts: %lu",
                 g_registered_count, g_total_packets, g_total_alerts);
        vTaskDelay(pdMS_TO_TICKS(15000));
    }
}
