// @inproceedings{lin2019tsm,
//   title={TSM: Temporal Shift Module for Efficient Video Understanding},
//   author={Lin, Ji and Gan, Chuang and Han, Song},
//   booktitle={Proceedings of the IEEE International Conference on Computer Vision},
//   year={2019}
// } 

#ifndef TSM_H
#define TSM_H

#include "darknet.h"
#include "dark_cuda.h"
#include "network.h"
#include "layer.h"
#include "utils.h"
#include "activations.h"
#include "convolutional_layer.h"
#include <stdio.h>  // srand
#include <stdlib.h>
#include "blas.h"
#include "gemm.h"

typedef layer tsm_layer;

#ifdef __cplusplus
extern "C" {
#endif // end cpp

layer make_tsm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, float partial_shift, int train);
void forward_tsm_layer(layer l, network_state state);
void backward_tsm_layer(layer l, network_state state);
void resize_tsm_layer(layer *l, int w, int h);

#ifdef GPU // GPU functions
void forward_tsm_layer_gpu(layer l, network_state state);
void backward_tsm_layer_gpu(layer l, network_state state);
void update_tsm_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_tsm_layer(layer l);
void pull_tsm_layer(layer l);
#endif // end GPU

#ifdef __cplusplus
}
#endif // end cpp
#endif // end TSM_H