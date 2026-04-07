/**
 * ═══════════════════════════════════════════════════════════════
 * TinyML Inference Engine — TFLite Micro on ESP32
 * ═══════════════════════════════════════════════════════════════
 * Loads the quantized border intrusion detection model and runs
 * inference on MFCC features extracted from the MEMS microphone.
 */

#include "tinyml_inference.h"
#include "config.h"
#include "esp_log.h"
#include "esp_timer.h"

/* TensorFlow Lite Micro headers */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Quantized model data (embedded as C array) */
#include "model_data.h"

static const char *TAG = "TINYML";

/* TFLite Micro components */
static const tflite::Model *g_model = nullptr;
static tflite::MicroInterpreter *g_interpreter = nullptr;
static TfLiteTensor *g_input_tensor = nullptr;
static TfLiteTensor *g_output_tensor = nullptr;

/* Tensor arena — allocated in PSRAM for larger models */
static uint8_t g_tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));


esp_err_t tinyml_init(void)
{
    ESP_LOGI(TAG, "Initializing TFLite Micro interpreter...");
    ESP_LOGI(TAG, "Tensor arena: %d bytes", TENSOR_ARENA_SIZE);
    
    /* Load model from embedded data */
    g_model = tflite::GetModel(g_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: %lu vs %d",
                 g_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }
    
    /* Register only the ops needed by our model */
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMean();          /* GlobalAveragePooling */
    resolver.AddMul();           /* BatchNormalization */
    resolver.AddAdd();           /* BatchNormalization */
    
    /* Build interpreter */
    static tflite::MicroInterpreter interpreter(
        g_model, resolver, g_tensor_arena, TENSOR_ARENA_SIZE
    );
    g_interpreter = &interpreter;
    
    /* Allocate tensors */
    TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return ESP_ERR_NO_MEM;
    }
    
    /* Cache input/output tensor pointers */
    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);
    
    ESP_LOGI(TAG, "Model loaded successfully:");
    ESP_LOGI(TAG, "  Input:  [%d, %d, %d, %d] (%s)",
             g_input_tensor->dims->data[0], g_input_tensor->dims->data[1],
             g_input_tensor->dims->data[2], g_input_tensor->dims->data[3],
             TfLiteTypeGetName(g_input_tensor->type));
    ESP_LOGI(TAG, "  Output: [%d, %d] (%s)",
             g_output_tensor->dims->data[0], g_output_tensor->dims->data[1],
             TfLiteTypeGetName(g_output_tensor->type));
    ESP_LOGI(TAG, "  Arena used: %d / %d bytes",
             g_interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
    
    return ESP_OK;
}


esp_err_t tinyml_invoke(const float *input_features, float *output_probs, 
                         int num_classes)
{
    if (!g_interpreter || !g_input_tensor || !g_output_tensor) {
        ESP_LOGE(TAG, "Interpreter not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    int64_t t_start = esp_timer_get_time();
    
    /* Copy input features to tensor */
    size_t input_size = g_input_tensor->bytes;
    
    if (g_input_tensor->type == kTfLiteFloat32) {
        memcpy(g_input_tensor->data.f, input_features, input_size);
    } else if (g_input_tensor->type == kTfLiteInt8) {
        /* Quantize input: q = (f / scale) + zero_point */
        float scale = g_input_tensor->params.scale;
        int32_t zero_point = g_input_tensor->params.zero_point;
        int8_t *q_data = g_input_tensor->data.int8;
        int num_elements = input_size;
        
        for (int i = 0; i < num_elements; i++) {
            int32_t q = (int32_t)roundf(input_features[i] / scale) + zero_point;
            q_data[i] = (int8_t)fmaxf(-128, fminf(127, q));
        }
    }
    
    /* Run inference */
    TfLiteStatus invoke_status = g_interpreter->Invoke();
    
    int64_t t_end = esp_timer_get_time();
    float elapsed_ms = (t_end - t_start) / 1000.0f;
    
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return ESP_FAIL;
    }
    
    /* Extract output probabilities */
    if (g_output_tensor->type == kTfLiteFloat32) {
        for (int i = 0; i < num_classes; i++) {
            output_probs[i] = g_output_tensor->data.f[i];
        }
    } else if (g_output_tensor->type == kTfLiteInt8) {
        float scale = g_output_tensor->params.scale;
        int32_t zero_point = g_output_tensor->params.zero_point;
        
        for (int i = 0; i < num_classes; i++) {
            output_probs[i] = (g_output_tensor->data.int8[i] - zero_point) * scale;
        }
    }
    
    ESP_LOGI(TAG, "Inference complete in %.1f ms", elapsed_ms);
    ESP_LOGI(TAG, "  Footstep: %.3f | Gunshot: %.3f | Noise: %.3f",
             output_probs[0], output_probs[1], output_probs[2]);
    
    return ESP_OK;
}


size_t tinyml_get_arena_size(void)
{
    return g_interpreter ? g_interpreter->arena_used_bytes() : 0;
}
