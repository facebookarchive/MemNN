// Copyright 2004-present Facebook. All Rights Reserved.
#include <math.h>

void addTFIDF(float* inp, float* out, int size, double tfpow, double* freqs) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        int i2 = i << 1;
        int ind = (int) inp[i];
        out[i2] = ind;
        float freq = freqs[ind - 1];
        float power = (tfpow > 0) ? 1 / pow(freq + 10, tfpow) : 1;
        out[i2 + 1] = power;
        sum += power * power;
    }
    float norm = sqrt(sum);
    if (norm > 0.00001) {
        for (int i = 1; i < size << 1; i += 2) {
            out[i] /= norm;
        }
    }
}

void resolveMinOcc(
    int dim, int dictMinOcc, long* sizes, float* x, double* index_to_freq
) {
    int i, d, word, size, data_len;
    size = sizes[0];
    data_len = 1;
    for (d = 1; d < dim; d++) {
        data_len *= sizes[d];
    }
    for (i = 0; i < size * data_len; i += data_len) {
        word = (int) x[i] - 1;
        if (index_to_freq[word] < dictMinOcc) {
            x[i] = 3;
        }
    }
}
