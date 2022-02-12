#include "tsm_layer.h"
#include "convolutional_layer.h"
#include "dark_cuda.h"

/*
Consists of a residual block:
- Temporal shift module
- Convolutional layer
*/
tsm_layer make_tsm_layer(int batch, int h, int w, int c, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, float partial_shift, int train){
    layer l = { (LAYER_TYPE)0 };

    int total_batch = batch*steps;

    l.w = w;
    l.h = h;
    l.c = c;
    l.out_c = 2*c;
    l.batch = batch;
    
    l.train = train;

    l.type = TSM;
    l.tsm_cache = (float*)xcalloc(batch * h * w * c, sizeof(float)); // residual features

    l.output_layer = (layer*)xcalloc(1, sizeof(layer));
    *(l.output_layer) = make_convolutional_layer(batch, steps, h, w, c, c, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, 0, 0, 0, 0, 0, NULL, 0, 0, train);
    l.output_layer->batch = batch;
    if (l.workspace_size < l.output_layer->workspace_size) l.workspace_size = l.output_layer->workspace_size;
    
    // out_h and out_w should be the same as input
    int out_h = convolutional_out_height(*(l.output_layer));
    int out_w = convolutional_out_width(*(l.output_layer));     
    l.out_w = out_w;
    l.out_h = out_h;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));

    l.forward = forward_tsm_layer;
    l.backward = backward_tsm_layer;

#ifdef GPU
    l.forward_gpu = forward_tsm_layer_gpu;
    l.backward_gpu = backward_tsm_layer_gpu;
    l.tsm_cache_gpu = cuda_make_array(l.tsm_cache, batch * h * w * c);
#endif
    return l;
}

/*
Concatenates y to the end of x along 0th dimension.
*/
void concat_channels(float *x, float *y, layer *l){
    int b, w, h, c;

    for(b = 0; b < l->batch; b++){
        for (w = 0; w < l->w; w++) {
            for (h = 0; h < l->h; h++) {
                for (c = 0; c < l->c; c++)
                {
                    int x_idx = w + l->w * (h + l->h * (c + l->c * b));
                    int y_idx = w + l->w * (h + l->h * (c + l->out_c * b)); // same, but shifted
                    l->output[x_idx] = x[x_idx];
                    l->output[y_idx] = y[x_idx];
                }
            }
        }
    }

}

/*
Perform random shift and update cache.
Note: may want to adopt a different shifting regime.
*/
void shift_and_update(float *frame, tsm_layer *l){
    int b, w, h, c;
    int len = l->batch * l->w * l->h * l->c;
    int* shift_ind = (int*)xcalloc(l->c, sizeof(int));

    // determines which channels to shift
    for(int i = 0; i < l->c; c++){
        shift_ind[i] = random_float() < 0.25;
    }

    for(b = 0; b < l->batch; b++){
        for (w = 0; w < l->w; w++) {
            for (h = 0; h < l->h; h++) {
                for (c = 0; c < l->c; c++)
                {
                    if(shift_ind[c]){
                        int idx = w + l->w * (h + l->h * (c + l->c * b));
                        frame[idx] = l->tsm_cache[idx]; // shift up
                    }
                }
            }
        }
    }

    float *p = (float*)xcalloc(len, sizeof(float));
    memcpy(p, frame, len * sizeof(float));
    l->tsm_cache = p; // making the modified input the input for next frame

    free(p);
    free(shift_ind);
}

void forward_tsm_layer(tsm_layer l, network_state state){
    // perform tsm with previous layer
    // perform convolution on it
    // return sum of output and input
    layer output_layer = *(l.output_layer);

    int len = l.batch * l.w * l.h * l.c;
    float *ipt = (float*)xcalloc(len, sizeof(float));
    memcpy(ipt, state.input, len * sizeof(float));

    // Shift
    shift_and_update(state.input, &l);
    // Run convolution
    forward_convolutional_layer(output_layer, state);
    // Concat ipt to state.input
    concat_channels(state.input, ipt, &l);
    
    free(ipt);
}

void backward_tsm_layer(layer l, network_state state){
    layer output_layer = *(l.output_layer);
    backward_convolutional_layer(output_layer, state);
}

void resize_tsm_layer(layer *l, int w, int h){
    l->w = w;
    l->h = h;
    l->inputs = h * w * l->c;
    resize_convolutional_layer(l->output_layer, w, h);
}

#ifdef GPU
void forward_tsm_layer_gpu(layer l, network_state state){
    // perform tsm with previous layer
    // perform convolution on it
    // return sum of output and input
    layer output_layer = *(l.output_layer);

    int len = l.batch * l.w * l.h * l.c;
    float *ipt = (float*)xcalloc(len, sizeof(float));
    memcpy(ipt, state.input, len * sizeof(float));

    // Shift
    shift_and_update(state.input, &l);
    // Run convolution
    forward_convolutional_layer_gpu(output_layer, state);
    // Concat ipt to state.input
    concat_channels(state.input, ipt, &l);
    
    free(ipt);
}

void backward_tsm_layer_gpu(layer l, network_state state){
    layer output_layer = *(l.output_layer);
    backward_convolutional_layer_gpu(output_layer, state);
}

void push_tsm_layer(layer l){
    push_convolutional_layer(*(l.output_layer));
}

void pull_tsm_layer(layer l){
    pull_convolutional_layer(*(l.output_layer));
}

void update_tsm_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale){
    update_convolutional_layer_gpu(*(l.output_layer), batch, learning_rate, momentum, decay, loss_scale);
}
#endif