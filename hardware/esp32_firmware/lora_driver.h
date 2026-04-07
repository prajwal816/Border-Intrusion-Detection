#ifndef __LORA_DRIVER_H__
#define __LORA_DRIVER_H__

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

void      lora_spi_init(void);
esp_err_t lora_init(void);
void      lora_set_frequency(long frequency);
void      lora_set_spreading_factor(int sf);
void      lora_set_bandwidth(long bandwidth);
void      lora_set_tx_power(int level);
esp_err_t lora_send_packet(const uint8_t *data, size_t length);
int       lora_receive_packet(uint8_t *buffer, size_t max_len);
float     lora_get_packet_rssi(void);
float     lora_get_packet_snr(void);
uint16_t  lora_compute_crc16(const uint8_t *data, size_t length);
float     read_internal_temperature(void);

#endif
