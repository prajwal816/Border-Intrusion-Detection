#ifndef __TINYML_INFERENCE_H__
#define __TINYML_INFERENCE_H__

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

esp_err_t tinyml_init(void);
esp_err_t tinyml_invoke(const float *input_features, float *output_probs, int num_classes);
size_t    tinyml_get_arena_size(void);

#endif
