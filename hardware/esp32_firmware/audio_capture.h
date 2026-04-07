#ifndef __AUDIO_CAPTURE_H__
#define __AUDIO_CAPTURE_H__

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

void  audio_i2s_init(void);
void  audio_capture_start(void);
void  audio_capture_stop(void);
float audio_compute_rms(const int16_t *buffer, size_t length);
void  audio_extract_mfcc(const int16_t *audio_buffer, size_t audio_length,
                          float *mfcc_output, int n_mfcc, int time_steps);

#endif
