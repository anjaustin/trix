/*
 * TriX Kernel Implementation
 * 
 * Sparse 2-bit ternary matrix multiplication with tile-based routing.
 * ARM NEON optimized for ternary weights {-1, 0, +1}.
 */

#include "trix.h"
#include <cstring>
#include <cmath>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

// Scalar dot product for packed ternary weights
static inline float dot_packed_scalar(
    const float* input,
    const uint8_t* packed_row,
    int in_features
) {
    float acc = 0.0f;
    int packed_len = (in_features + 3) / 4;
    
    for (int p = 0; p < packed_len; p++) {
        uint8_t packed = packed_row[p];
        int base = p * 4;
        
        for (int i = 0; i < 4 && (base + i) < in_features; i++) {
            uint8_t code = (packed >> (i * 2)) & 0x03;
            if (code == 0x01) {
                acc += input[base + i];  // +1
            } else if (code == 0x02) {
                acc -= input[base + i];  // -1
            }
        }
    }
    return acc;
}

#if USE_NEON
// NEON-optimized dot product for packed ternary weights
static inline float dot_packed_neon(
    const float* input,
    const uint8_t* packed_row,
    int in_features
) {
    float32x4_t acc_vec = vdupq_n_f32(0.0f);
    int packed_len = (in_features + 3) / 4;
    int full_packs = in_features / 4;
    
    for (int p = 0; p < full_packs; p++) {
        uint8_t packed = packed_row[p];
        float32x4_t in_vec = vld1q_f32(&input[p * 4]);
        
        float weights[4];
        for (int i = 0; i < 4; i++) {
            uint8_t code = (packed >> (i * 2)) & 0x03;
            weights[i] = (code == 0x01) ? 1.0f : (code == 0x02) ? -1.0f : 0.0f;
        }
        float32x4_t w_vec = vld1q_f32(weights);
        acc_vec = vmlaq_f32(acc_vec, in_vec, w_vec);
    }
    
    // Horizontal sum
    float32x2_t sum = vadd_f32(vget_low_f32(acc_vec), vget_high_f32(acc_vec));
    sum = vpadd_f32(sum, sum);
    float acc = vget_lane_f32(sum, 0);
    
    // Handle remainder
    int remaining_start = full_packs * 4;
    if (remaining_start < in_features) {
        uint8_t packed = packed_row[full_packs];
        for (int i = 0; remaining_start + i < in_features; i++) {
            uint8_t code = (packed >> (i * 2)) & 0x03;
            if (code == 0x01) {
                acc += input[remaining_start + i];
            } else if (code == 0x02) {
                acc -= input[remaining_start + i];
            }
        }
    }
    
    return acc;
}
#endif

// Dispatch to best available implementation
static inline float dot_packed(
    const float* input,
    const uint8_t* packed_row,
    int in_features
) {
#if USE_NEON
    return dot_packed_neon(input, packed_row, in_features);
#else
    return dot_packed_scalar(input, packed_row, in_features);
#endif
}

extern "C" void trix_forward(
    const float* input,
    const uint8_t* packed_w,
    const float* scales,
    const int8_t* gate_mask,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
) {
    int out_per_tile = out_features / num_tiles;
    int packed_in_dim = (in_features + 3) / 4;
    
    memset(output, 0, batch_size * out_features * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        const float* in_row = &input[b * in_features];
        float* out_row = &output[b * out_features];
        
        for (int t = 0; t < num_tiles; t++) {
            // Skip inactive tiles - this is where the speedup comes from
            if (gate_mask[b * num_tiles + t] == 0) {
                continue;
            }
            
            int out_start = t * out_per_tile;
            const uint8_t* tile_weights = &packed_w[t * out_per_tile * packed_in_dim];
            const float* tile_scales = &scales[t * out_per_tile];
            
            for (int o = 0; o < out_per_tile; o++) {
                const uint8_t* w_row = &tile_weights[o * packed_in_dim];
                float dot = dot_packed(in_row, w_row, in_features);
                out_row[out_start + o] = dot * tile_scales[o];
            }
        }
    }
}

extern "C" void pack_weights(
    const float* raw_weights,
    uint8_t* packed_output,
    int rows,
    int cols
) {
    int packed_cols = (cols + 3) / 4;
    
    for (int r = 0; r < rows; r++) {
        for (int pc = 0; pc < packed_cols; pc++) {
            uint8_t packed = 0;
            for (int i = 0; i < 4; i++) {
                int c = pc * 4 + i;
                uint8_t code = 0x00;
                if (c < cols) {
                    float w = raw_weights[r * cols + c];
                    if (w > 0.5f) {
                        code = 0x01;  // +1
                    } else if (w < -0.5f) {
                        code = 0x02;  // -1
                    }
                }
                packed |= (code << (i * 2));
            }
            packed_output[r * packed_cols + pc] = packed;
        }
    }
}

extern "C" void unpack_weights(
    const uint8_t* packed_weights,
    float* raw_output,
    int rows,
    int cols
) {
    int packed_cols = (cols + 3) / 4;
    
    for (int r = 0; r < rows; r++) {
        for (int pc = 0; pc < packed_cols; pc++) {
            uint8_t packed = packed_weights[r * packed_cols + pc];
            for (int i = 0; i < 4; i++) {
                int c = pc * 4 + i;
                if (c < cols) {
                    uint8_t code = (packed >> (i * 2)) & 0x03;
                    float val = 0.0f;
                    if (code == 0x01) val = 1.0f;
                    else if (code == 0x02) val = -1.0f;
                    raw_output[r * cols + c] = val;
                }
            }
        }
    }
}
