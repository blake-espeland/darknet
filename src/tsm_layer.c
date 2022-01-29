#include "tsm_layer.h"

layer make_tsm_layer(int batch, int t, int w, int h, int c, int reverse){
    layer l = { (LAYER_TYPE)0 };
    l.t = batch;
    l.w = w;
    l.h = h;
    l.batch = batch;
    l.type = TSM;
    l.tsm_cache = (float*)xcalloc(1, sizeof(layer)); // previous layer
    
}

// class TemporalShift(nn.Module):
//     def __init__(self, n_segment=3, n_div=8, inplace=False):
//         super(TemporalShift, self).__init__()
//         self.n_segment = n_segment
//         self.fold_div = n_div
//         self.inplace = inplace
//         if inplace:
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