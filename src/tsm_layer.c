#include <stdio.h>  // srand
#include <stdlib.h>

#include "tsm_layer.h"
#include "utils.h"

/*
Only works if in video training mode: input (n, t, h, w, c)
The t dimension is only shared with this layer.
Consists of a residual block:
- Temporal shift module
- Convolutional layer
*/
layer make_tsm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, float partial_shift, int train){
    layer l = { (LAYER_TYPE)0 };
    l.w = w;
    l.h = h;
    l.c = c;
    l.batch = batch;
    l.type = TSM;
    l.tsm_cache = (float*)xcalloc(1, sizeof(float)); // residual features

    l.output_layer = (layer*)xcalloc(1, sizeof(layer));
    *(l.output_layer) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, 0, 0, 0, 0, 0, NULL, 0, 0, train);
    l.output_layer->batch = batch;
    if (l.workspace_size < l.output_layer->workspace_size) l.workspace_size = l.output_layer->workspace_size;
}

/*
Perform random shift and update cache
*/
void shift(float *frame, const tsm_layer *l){
    int b;
    int w, h, c;
    int len = l->batch * l->w * l->h * l->c;
    int* shift_ind = (int*)xcalloc(l->c, sizeof(int));

    // determines which channels to shift
    for(int i = 0; i < l->c; c++){
        shift_ind[i] = random_float() < 0.25;
    }

    for(int b = 0; b < l->batch; b++){
        for (w = 0; w < l->w; w++) {
            for (h = 0; h < l->h; h++) {
                for (c = 0; c < l->c; c++)
                {
                    if(shift_ind[c]){
                        int idx = w + l->w * (h + l->h * (c + l->c * b));
                        frame[idx] = l->tsm_cache[idx]; // shift down
                    }
                }
            }
        }
    }
    float *p = (float*)xcalloc(len, sizeof(float));
    memcpy(p, frame, len * sizeof(float));
    l->tsm_cache = p;

    free(p);
    free(shift_ind);
}

void forward_tsm_layer(const tsm_layer l, network_state state){
    // perform tsm with previous layer
    // perform convolution on it
    // return sum of output and input
    int len = l.batch * l.w * l.h * l.c;
    float *p = (float*)xcalloc(len, sizeof(float));
    memcpy(p, state.input, len * sizeof(float));

    l->tsm_cache = p;
    shift(state.input, &l);
    
    free(p);
}



// class TemporalShift(nn.Module):
//     def __init__(self, n_segment=3, n_div=8, inplace=False):
//         super(TemporalShift, self).__init__()
//         self.n_segment = n_segment
//         self.fold_div = n_div
//         self.inplace = inplace
//         if inplace:`
//             print('=> Using in-place shift...')
//         print('=> Using fold div: {}'.format(self.fold_div))

//     def forward(self, x):
//         return self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace) 

//     @staticmethod
//     def shift(x, n_segment, fold_div=3, ):
//         nt, c, h, w = x.size()
//         n_batch = nt // n_segment
//         x = x.view(n_batch, n_segment, c, h, w)

//         fold = c // fold_div

//         out = torch.zeros_like(x)
//         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
//         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
//         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

//         return out.view(nt, c, h, w)