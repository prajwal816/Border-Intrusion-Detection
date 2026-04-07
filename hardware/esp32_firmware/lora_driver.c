/**
 * ═══════════════════════════════════════════════════════════════
 * LoRa SX1276 Radio Driver — SPI Communication
 * ═══════════════════════════════════════════════════════════════
 * Low-level driver for the Semtech SX1276 LoRa transceiver
 * connected to ESP32 via HSPI bus.
 */

#include "lora_driver.h"
#include "config.h"
#include <string.h>
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "LORA_DRV";

/* SX1276 Register Addresses */
#define REG_FIFO                0x00
#define REG_OP_MODE             0x01
#define REG_FRF_MSB             0x06
#define REG_FRF_MID             0x07
#define REG_FRF_LSB             0x08
#define REG_PA_CONFIG           0x09
#define REG_PA_RAMP             0x0A
#define REG_OCP                 0x0B
#define REG_LNA                 0x0C
#define REG_FIFO_ADDR_PTR       0x0D
#define REG_FIFO_TX_BASE        0x0E
#define REG_FIFO_RX_BASE        0x0F
#define REG_IRQ_FLAGS           0x12
#define REG_RX_NB_BYTES         0x13
#define REG_MODEM_STAT          0x18
#define REG_PKT_SNR             0x19
#define REG_PKT_RSSI            0x1A
#define REG_RSSI                0x1B
#define REG_MODEM_CONFIG_1      0x1D
#define REG_MODEM_CONFIG_2      0x1E
#define REG_SYMB_TIMEOUT_LSB    0x1F
#define REG_PREAMBLE_MSB        0x20
#define REG_PREAMBLE_LSB        0x21
#define REG_PAYLOAD_LENGTH      0x22
#define REG_MODEM_CONFIG_3      0x26
#define REG_FIFO_RX_BYTE_ADDR   0x25
#define REG_DIO_MAPPING_1       0x40
#define REG_VERSION             0x42
#define REG_PA_DAC              0x4D

/* Operating modes */
#define MODE_SLEEP              0x00
#define MODE_STANDBY            0x01
#define MODE_TX                 0x03
#define MODE_RX_CONTINUOUS      0x05
#define MODE_LORA               0x80

/* IRQ Flags */
#define IRQ_TX_DONE             0x08
#define IRQ_RX_DONE             0x40
#define IRQ_PAYLOAD_CRC_ERROR   0x20

static spi_device_handle_t g_spi_handle;
static bool g_initialized = false;


/* ─── SPI Helper Functions ─── */

static uint8_t _spi_read_register(uint8_t addr)
{
    spi_transaction_t t;
    memset(&t, 0, sizeof(t));
    t.length = 16;
    uint8_t tx_data[2] = {addr & 0x7F, 0x00};
    uint8_t rx_data[2] = {0};
    t.tx_buffer = tx_data;
    t.rx_buffer = rx_data;
    spi_device_transmit(g_spi_handle, &t);
    return rx_data[1];
}

static void _spi_write_register(uint8_t addr, uint8_t value)
{
    spi_transaction_t t;
    memset(&t, 0, sizeof(t));
    t.length = 16;
    uint8_t tx_data[2] = {addr | 0x80, value};
    t.tx_buffer = tx_data;
    spi_device_transmit(g_spi_handle, &t);
}


/* ─── Initialization ─── */

void lora_spi_init(void)
{
    spi_bus_config_t bus_cfg = {
        .miso_io_num = LORA_MISO_PIN,
        .mosi_io_num = LORA_MOSI_PIN,
        .sclk_io_num = LORA_SCK_PIN,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 256,
    };
    
    spi_device_interface_config_t dev_cfg = {
        .clock_speed_hz = 8 * 1000 * 1000,  /* 8 MHz */
        .mode = 0,
        .spics_io_num = LORA_CS_PIN,
        .queue_size = 1,
    };
    
    ESP_ERROR_CHECK(spi_bus_initialize(LORA_SPI_HOST, &bus_cfg, SPI_DMA_CH_AUTO));
    ESP_ERROR_CHECK(spi_bus_add_device(LORA_SPI_HOST, &dev_cfg, &g_spi_handle));
    
    /* Reset SX1276 */
    gpio_set_direction(LORA_RST_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LORA_RST_PIN, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(LORA_RST_PIN, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
    
    ESP_LOGI(TAG, "SPI bus initialized for SX1276");
}


esp_err_t lora_init(void)
{
    /* Verify chip version */
    uint8_t version = _spi_read_register(REG_VERSION);
    if (version != 0x12) {
        ESP_LOGE(TAG, "SX1276 not found (version: 0x%02X)", version);
        return ESP_ERR_NOT_FOUND;
    }
    ESP_LOGI(TAG, "SX1276 detected (version: 0x%02X)", version);
    
    /* Set sleep mode */
    _spi_write_register(REG_OP_MODE, MODE_SLEEP | MODE_LORA);
    vTaskDelay(pdMS_TO_TICKS(10));
    
    /* Set FIFO base addresses */
    _spi_write_register(REG_FIFO_TX_BASE, 0x00);
    _spi_write_register(REG_FIFO_RX_BASE, 0x00);
    
    /* Set LNA boost */
    _spi_write_register(REG_LNA, _spi_read_register(REG_LNA) | 0x03);
    
    /* Auto AGC */
    _spi_write_register(REG_MODEM_CONFIG_3, 0x04);
    
    /* Standby mode */
    _spi_write_register(REG_OP_MODE, MODE_STANDBY | MODE_LORA);
    
    g_initialized = true;
    ESP_LOGI(TAG, "SX1276 LoRa initialized");
    
    return ESP_OK;
}


void lora_set_frequency(long frequency)
{
    uint64_t frf = ((uint64_t)frequency << 19) / 32000000;
    _spi_write_register(REG_FRF_MSB, (uint8_t)(frf >> 16));
    _spi_write_register(REG_FRF_MID, (uint8_t)(frf >> 8));
    _spi_write_register(REG_FRF_LSB, (uint8_t)(frf >> 0));
    ESP_LOGI(TAG, "Frequency set: %ld Hz", frequency);
}


void lora_set_spreading_factor(int sf)
{
    if (sf < 6) sf = 6;
    if (sf > 12) sf = 12;
    
    uint8_t reg = _spi_read_register(REG_MODEM_CONFIG_2);
    _spi_write_register(REG_MODEM_CONFIG_2, (reg & 0x0F) | ((sf << 4) & 0xF0));
    
    if (sf == 6) {
        _spi_write_register(0x31, 0xC5);
        _spi_write_register(0x37, 0x0C);
    }
    
    ESP_LOGI(TAG, "Spreading factor: SF%d", sf);
}


void lora_set_bandwidth(long bandwidth)
{
    int bw_code;
    if      (bandwidth <= 7800)   bw_code = 0;
    else if (bandwidth <= 10400)  bw_code = 1;
    else if (bandwidth <= 15600)  bw_code = 2;
    else if (bandwidth <= 20800)  bw_code = 3;
    else if (bandwidth <= 31250)  bw_code = 4;
    else if (bandwidth <= 41700)  bw_code = 5;
    else if (bandwidth <= 62500)  bw_code = 6;
    else if (bandwidth <= 125000) bw_code = 7;
    else if (bandwidth <= 250000) bw_code = 8;
    else                          bw_code = 9;
    
    uint8_t reg = _spi_read_register(REG_MODEM_CONFIG_1);
    _spi_write_register(REG_MODEM_CONFIG_1, (reg & 0x0F) | (bw_code << 4));
    
    ESP_LOGI(TAG, "Bandwidth: %ld Hz (code: %d)", bandwidth, bw_code);
}


void lora_set_tx_power(int level)
{
    if (level > 17) {
        _spi_write_register(REG_PA_DAC, 0x87);
        level = (level < 20) ? level : 20;
        _spi_write_register(REG_PA_CONFIG, 0x80 | (level - 5));
    } else {
        _spi_write_register(REG_PA_DAC, 0x84);
        level = (level < 2) ? 2 : level;
        _spi_write_register(REG_PA_CONFIG, 0x80 | (level - 2));
    }
    
    ESP_LOGI(TAG, "TX power: %d dBm", level);
}


esp_err_t lora_send_packet(const uint8_t *data, size_t length)
{
    if (!g_initialized) return ESP_ERR_INVALID_STATE;
    if (length > 255) return ESP_ERR_INVALID_SIZE;
    
    /* Standby mode */
    _spi_write_register(REG_OP_MODE, MODE_STANDBY | MODE_LORA);
    
    /* Set FIFO pointer */
    _spi_write_register(REG_FIFO_ADDR_PTR, 0x00);
    
    /* Write payload to FIFO */
    for (size_t i = 0; i < length; i++) {
        _spi_write_register(REG_FIFO, data[i]);
    }
    
    /* Set payload length */
    _spi_write_register(REG_PAYLOAD_LENGTH, length);
    
    /* Start TX */
    _spi_write_register(REG_OP_MODE, MODE_TX | MODE_LORA);
    
    /* Wait for TX done (polling DIO0) */
    int timeout = 5000;  /* 5s max */
    while (timeout-- > 0) {
        uint8_t irq = _spi_read_register(REG_IRQ_FLAGS);
        if (irq & IRQ_TX_DONE) {
            _spi_write_register(REG_IRQ_FLAGS, IRQ_TX_DONE);
            ESP_LOGI(TAG, "Packet transmitted: %d bytes", length);
            return ESP_OK;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    ESP_LOGE(TAG, "TX timeout");
    return ESP_ERR_TIMEOUT;
}


uint16_t lora_compute_crc16(const uint8_t *data, size_t length)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) crc = (crc >> 1) ^ 0xA001;
            else         crc >>= 1;
        }
    }
    return crc;
}


float read_internal_temperature(void)
{
    /* ESP32 internal temperature sensor */
    uint8_t raw = 0;
    /* Simplified — actual implementation uses TSENS peripheral */
    return 25.0f + (raw - 128) * 0.4f;
}
