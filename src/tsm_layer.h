#ifndef TSM_H
#define TSM_H
#include "darknet.h"
#include "network.h"
#include "layer.h"
#include "utils.h"
#include "convolutional_layer.h"

typedef layer tsm_layer;

/*
Reorg features (opt = 0.25)
Store feature cache

*/

#ifdef __cplusplus
extern "C" {
#endif // end cpp

layer make_tsm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, float partial_shift, int train);
void cache_tsm_features(layer *l, int w, int h);
void forward_tsm_layer(const layer l, network_state state);
void backward_tsm_layer(const layer l, network_state state);

// void shift(float *frame, const tsm_layer *l);

#ifdef GPU // GPU functions
void forward_tsm_layer_gpu(layer l, network_state state);
void backward_tsm_layer_gpu(layer l, network_state state);
#endif // end GPU

#ifdef __cplusplus
}
#endif // end cpp
#endif // end TSM_H