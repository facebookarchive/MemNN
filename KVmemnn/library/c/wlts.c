// Copyright 2004-present Facebook. All Rights Reserved.
#include <math.h>
#include <stdio.h>

void wlts_updateOutput(
    int dims, long* sizes, int input_dim,
    float* input, float* weight, float* output
) {
    int i, d, k, j, i2, word, wgt_off, out_off, data_len;
    float coeff;
    int size = sizes[0];
    data_len = 1;
    for (d = 1; d < dims; d++) {
        data_len *= sizes[d];
    }
    if (input_dim == 2) {
        /* normal setting: input = (word_index, weight) pairs */
        for (i = 0; i < size; i++) {
            i2 = i << 1;
            word = (int) input[i2] - 1;
            coeff = input[i2 + 1];
            out_off = i * data_len;
            wgt_off = word * data_len;
            for (j = 0; j < data_len; j++) {
                output[out_off + j] = weight[wgt_off + j] * coeff;
            }
        }
    } else {
        /* no weight for each word_index */
        for (i = 0; i < size; i++) {
            word = (int) input[i] - 1;
            out_off = i * data_len;
            wgt_off = word * data_len;
            for (j = 0; j < data_len; j++) {
                output[out_off + j] = weight[wgt_off + j];
            }
        }
    }
}

int wlts_accGradParameters(
    float scale, int size, int dims, long* sizes, int input_dim, long* inputs,
    float* input, float* gradOutput, float* gradWeight, int* inputWeights
) {
    int i, i_dim, d, j, k, index, ind_off, data_len, grd_wgt_off, grd_out_off;
    float coeff;
    int gradWeight_size = sizes[0];
    data_len = 1;
    for (d = 1; d < dims; d++) {
        data_len *= sizes[d];
    }
    for (i = 0; i < size; i++) {
        i_dim = i * input_dim;
        k = (int) input[i_dim];
        if (inputs[0] < gradWeight_size) {
            ind_off = (int) inputs[0];
            index = ind_off + 1;
            inputs[0] = (long) index;
            inputWeights[ind_off] = k;
            grd_wgt_off = ind_off * data_len;
            grd_out_off = i * data_len;
            if (input_dim == 2) {
                /* normal setting: input = (word_index, weight) pairs */
                coeff = scale * input[i_dim + 1];
            } else {
                /* no weight for each word_index */
                coeff = scale;
            }
            for (j = 0; j < data_len; j++) {
                gradWeight[grd_wgt_off + j] =
                    coeff * gradOutput[grd_out_off + j];
            }

        } else {
            // break and request resize
            return 1;
        }
    }
    // return success
    return 0;
}

void wlts_accUpdateGradParameters(
    float lr, int size, int dims, long* sizes, int input_dim,
    float* input, float* gradOutput, float* weight
) {
    int i, d, j, i2, word, wgt_off, grd_off, data_len;
    float coeff;
    data_len = 1;
    for (d = 1; d < dims; d++) {
        data_len *= sizes[d];
    }
    for (i = 0; i < size; i++) {
        if (input_dim == 2) {
            /* normal setting: input = (word_index, weight) pairs */
            i2 = i << 1;
            word = (int) input[i2] - 1;
            coeff = lr * input[i2 + 1];
        } else {
            /* no weight for each word_index */
            word = (int) input[i] - 1;
            coeff = lr;
        }
        grd_off = i * data_len;
        wgt_off = word * data_len;
        for (j = 0; j < data_len; j++) {
            weight[wgt_off + j] -= coeff * gradOutput[grd_off + j];
        }
    }
}

void wlts_updateParameters(
    float lr, long inputs, int dims, long* sizes,
    float* weight, float* gradWeight, int* inputWeights
) {
    int data_len = 1;
    for (int d = 1; d < dims; d++) {
        data_len *= sizes[d];
    }
    for (int i = 0; i < inputs; i++) {
        int word = inputWeights[i] - 1;
        int word_off = word * data_len;
        int grd_wgt_off = i * data_len;
        for (int j = 0; j < data_len; j++) {
            weight[word_off + j] -= lr * gradWeight[grd_wgt_off + j];
        }
    }
}
