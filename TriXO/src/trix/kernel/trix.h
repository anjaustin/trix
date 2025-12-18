/*
 * TriX Kernel Header
 * 
 * Sparse 2-bit ternary matrix multiplication with tile-based routing.
 * Implements ARM NEON acceleration for ternary weights {-1, 0, +1}.
 */

#ifndef TRIX_H
#define TRIX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Forward pass with sparse 2-bit weights.
 *
 * Args:
 *   input: Input tensor [batch_size, in_features]
 *   packed_w: Packed 2-bit weights [num_tiles * out_per_tile, packed_in_dim]
 *   scales: Per-output scaling factors [out_features]
 *   gate_mask: Tile activation mask [batch_size, num_tiles]
 *   output: Output tensor [batch_size, out_features]
 *   batch_size: Batch dimension
 *   in_features: Input dimension
 *   out_features: Output dimension
 *   num_tiles: Number of routing tiles
 */
void trix_forward(
    const float* input,
    const uint8_t* packed_w,
    const float* scales,
    const int8_t* gate_mask,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
);

/*
 * Pack float32 ternary weights to 2-bit representation.
 * 
 * Encoding: +1 -> 0b01, -1 -> 0b10, 0 -> 0b00
 * 4 weights pack into 1 byte (16x compression vs float32).
 *
 * Args:
 *   raw_weights: Float weights with values in {-1, 0, +1}
 *   packed_output: Output buffer for packed weights
 *   rows: Number of output features
 *   cols: Number of input features
 */
void pack_weights(
    const float* raw_weights,
    uint8_t* packed_output,
    int rows,
    int cols
);

/*
 * Unpack 2-bit weights back to float32.
 *
 * Args:
 *   packed_weights: Packed 2-bit weights
 *   raw_output: Output buffer for float weights
 *   rows: Number of output features
 *   cols: Number of input features
 */
void unpack_weights(
    const uint8_t* packed_weights,
    float* raw_output,
    int rows,
    int cols
);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_H */
