/**
 * ═══════════════════════════════════════════════════════════════
 * Audio Capture Driver — ICS-43434 MEMS Microphone via I2S
 * ═══════════════════════════════════════════════════════════════
 */

#include "audio_capture.h"
#include "config.h"
#include <math.h>
#include <string.h>
#include "driver/i2s.h"
#include "esp_log.h"
#include "esp_dsp.h"

static const char *TAG = "AUDIO_DRV";

/* Hann window for FFT */
static float g_hann_window[FFT_SIZE];
static float g_mel_filterbank[N_MELS][FFT_SIZE / 2 + 1];
static bool  g_initialized = false;


void audio_i2s_init(void)
{
    i2s_config_t i2s_config = {
        .mode = I2S_MODE_MASTER | I2S_MODE_RX,
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .dma_buf_count = I2S_DMA_BUF_COUNT,
        .dma_buf_len = I2S_DMA_BUF_LEN,
        .use_apll = true,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num   = I2S_BCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DATA_IN_PIN,
    };
    
    ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT_NUM, &i2s_config, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT_NUM, &pin_config));
    ESP_ERROR_CHECK(i2s_set_clk(I2S_PORT_NUM, SAMPLE_RATE, 
                                 I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO));
    
    /* Pre-compute Hann window */
    for (int i = 0; i < FFT_SIZE; i++) {
        g_hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FFT_SIZE - 1)));
    }
    
    /* Pre-compute Mel filterbank */
    _compute_mel_filterbank();
    
    g_initialized = true;
    ESP_LOGI(TAG, "I2S initialized — SR: %d Hz, %d-bit, DMA: %dx%d",
             SAMPLE_RATE, AUDIO_BIT_DEPTH, I2S_DMA_BUF_COUNT, I2S_DMA_BUF_LEN);
}


void audio_capture_start(void)
{
    i2s_start(I2S_PORT_NUM);
}


void audio_capture_stop(void)
{
    i2s_stop(I2S_PORT_NUM);
}


float audio_compute_rms(const int16_t *buffer, size_t length)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < length; i++) {
        float sample = buffer[i] / 32768.0f;
        sum_sq += sample * sample;
    }
    return sqrtf(sum_sq / length);
}


/* ─── MFCC Extraction Pipeline ─── */

static void _compute_mel_filterbank(void)
{
    /* Convert Hz to Mel scale */
    float mel_low = 2595.0f * log10f(1.0f + 0.0f / 700.0f);
    float mel_high = 2595.0f * log10f(1.0f + (SAMPLE_RATE / 2.0f) / 700.0f);
    
    float mel_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        float mel = mel_low + (mel_high - mel_low) * i / (N_MELS + 1);
        mel_points[i] = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    
    /* Convert Mel points to FFT bin indices */
    int bin_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        bin_points[i] = (int)floorf((FFT_SIZE + 1) * mel_points[i] / SAMPLE_RATE);
    }
    
    /* Create triangular filters */
    memset(g_mel_filterbank, 0, sizeof(g_mel_filterbank));
    for (int m = 0; m < N_MELS; m++) {
        for (int k = bin_points[m]; k < bin_points[m + 1]; k++) {
            if (k < FFT_SIZE / 2 + 1) {
                g_mel_filterbank[m][k] = (float)(k - bin_points[m]) / 
                                         (bin_points[m + 1] - bin_points[m]);
            }
        }
        for (int k = bin_points[m + 1]; k < bin_points[m + 2]; k++) {
            if (k < FFT_SIZE / 2 + 1) {
                g_mel_filterbank[m][k] = (float)(bin_points[m + 2] - k) / 
                                         (bin_points[m + 2] - bin_points[m + 1]);
            }
        }
    }
    
    ESP_LOGI(TAG, "Mel filterbank computed: %d filters", N_MELS);
}


void audio_extract_mfcc(const int16_t *audio_buffer, size_t audio_length,
                         float *mfcc_output, int n_mfcc, int time_steps)
{
    float frame[FFT_SIZE];
    float spectrum[FFT_SIZE];
    float mel_energies[N_MELS];
    float dct_basis[N_MFCC][N_MELS];
    
    /* Pre-compute DCT-II basis */
    for (int i = 0; i < n_mfcc; i++) {
        for (int j = 0; j < N_MELS; j++) {
            dct_basis[i][j] = cosf(M_PI * i * (2 * j + 1) / (2.0f * N_MELS));
        }
    }
    
    int frame_count = 0;
    int hop = HOP_LENGTH;
    
    for (int t = 0; t < time_steps && (t * hop + FFT_SIZE) <= audio_length; t++) {
        /* Extract frame and apply window */
        for (int i = 0; i < FFT_SIZE; i++) {
            int idx = t * hop + i;
            frame[i] = (audio_buffer[idx] / 32768.0f) * g_hann_window[i];
        }
        
        /* FFT (using ESP-DSP hardware acceleration) */
        dsps_fft2r_fc32(frame, FFT_SIZE >> 1);
        dsps_bit_rev_fc32(frame, FFT_SIZE >> 1);
        
        /* Power spectrum */
        for (int i = 0; i <= FFT_SIZE / 2; i++) {
            float re = frame[2 * i];
            float im = (i < FFT_SIZE / 2) ? frame[2 * i + 1] : 0;
            spectrum[i] = re * re + im * im;
        }
        
        /* Apply Mel filterbank */
        for (int m = 0; m < N_MELS; m++) {
            mel_energies[m] = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; k++) {
                mel_energies[m] += spectrum[k] * g_mel_filterbank[m][k];
            }
            /* Log compression */
            mel_energies[m] = logf(fmaxf(mel_energies[m], 1e-10f));
        }
        
        /* DCT to get MFCCs */
        for (int i = 0; i < n_mfcc; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N_MELS; j++) {
                sum += mel_energies[j] * dct_basis[i][j];
            }
            mfcc_output[i * time_steps + t] = sum;
        }
        
        frame_count++;
    }
    
    /* Normalize MFCCs */
    float mean = 0, std = 0;
    int total = n_mfcc * frame_count;
    for (int i = 0; i < total; i++) mean += mfcc_output[i];
    mean /= total;
    for (int i = 0; i < total; i++) std += (mfcc_output[i] - mean) * (mfcc_output[i] - mean);
    std = sqrtf(std / total + 1e-8f);
    for (int i = 0; i < total; i++) mfcc_output[i] = (mfcc_output[i] - mean) / std;
    
    ESP_LOGI(TAG, "MFCC extracted: %d frames × %d coefficients", frame_count, n_mfcc);
}
