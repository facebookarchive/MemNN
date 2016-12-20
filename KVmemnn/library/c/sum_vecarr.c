// Copyright 2004-present Facebook. All Rights Reserved.

/**
 * Calculates the sum of each example in data, with lengths specified by len.
 */
void sum_doubles(double* data, float* len, int size, int dim, double* result) {
    int s, i, d, add, mult, offset;
    double sum[dim];
    offset = 0;
    // break out small cases--not looping over d is much faster for low dims
    if (dim == 1) {
        for (s = 0; s < size; s++) {
            sum[0] = 0;
            for (i = 0; i < len[s]; i++) {
                sum[0] += data[offset + i];
            }
            offset += i;
            result[s] = sum[0];
        }
    } else if (dim == 2) {
        for (s = 0; s < size; s++) {
            sum[0] = 0;
            sum[1] = 0;
            for (i = 0; i < ((int)len[s]) << 1; i += 2) {
                add = offset + i;
                sum[0] += data[add];
                sum[1] += data[add + 1];
            }
            offset += i;
            mult = s << 1;
            result[mult] = sum[0];
            result[mult + 1] = sum[1];
        }
    } else {
        for (s = 0; s < size; s++) {
            for (d = 0; d < dim; d++)
                sum[d] = 0;
            for (i = 0; i < len[s] * dim; i += dim) {
                add = offset + i;
                sum[0] += data[add];
                for (d = 1; d < dim; d++)
                    sum[d] += data[add + d];
            }
            offset += i;
            mult = s * dim;
            result[mult] = sum[0];
            for (d = 1; d < dim; d++)
                result[mult + d] = sum[d];
        }
    }
}

/**
 * Calculates the sum of each example in data, with lengths specified by len.
 */
void sum_floats(float* data, float* len, int size, int dim, float* result) {
    int s, i, d, add, mult, offset;
    double sum[dim];
    offset = 0;
    // break out small cases--not looping over d is much faster for low dims
    if (dim == 1) {
        for (s = 0; s < size; s++) {
            sum[0] = 0;
            for (i = 0; i < len[s]; i++) {
                sum[0] += data[offset + i];
            }
            offset += i;
            result[s] = sum[0];
        }
    } else if (dim == 2) {
        for (s = 0; s < size; s++) {
            sum[0] = 0;
            sum[1] = 0;
            for (i = 0; i < ((int)len[s]) << 1; i += 2) {
                add = offset + i;
                sum[0] += data[add];
                sum[1] += data[add + 1];
            }
            offset += i;
            mult = s << 1;
            result[mult] = sum[0];
            result[mult + 1] = sum[1];
        }
    } else {
        for (s = 0; s < size; s++) {
            for (d = 0; d < dim; d++)
                sum[d] = 0;
            for (i = 0; i < len[s] * dim; i += dim) {
                add = offset + i;
                sum[0] += data[add];
                for (d = 1; d < dim; d++)
                    sum[d] += data[add + d];
            }
            offset += i;
            mult = s * dim;
            result[mult] = sum[0];
            for (d = 1; d < dim; d++)
                result[mult + d] = sum[d];
        }
    }
}

/*
 * For each entry gradoutput[o], copy it len[o] times into gradinput.
 */
void grad_doubles(
    double* gradOutput, float* len, int outputSize, int dim, double* gradInput
) {
    int o, i, d, mult, offset;
    i = 0;
    offset = 0;
    // break out small cases--not looping over d is much faster for low dims
    if (dim == 1) {
        for (o = 0; o < outputSize; o++) {
            for (; i < offset + len[o]; i++)
                gradInput[i] = gradOutput[o];
            offset = i;
        }
    } else if (dim == 2) {
        for (o = 0; o < outputSize; o++) {
            for (; i < offset + len[o] * 2; i += 2) {
                mult = o * 2;
                gradInput[i] = gradOutput[mult];
                gradInput[i + 1] = gradOutput[mult + 1];
            }
            offset = i;
        }
    } else {
        for (o = 0; o < outputSize; o++) {
            for (; i < offset + len[o] * dim; i += dim) {
                mult = o * dim;
                gradInput[i] = gradOutput[mult];
                for (d = 1; d < dim; d++)
                    gradInput[i + d] = gradOutput[mult + d];
            }
            offset = i;
        }
    }
}

/*
 * For each entry gradoutput[o], copy it len[o] times into gradinput.
 */
void grad_floats(
    float* gradOutput, float* len, int outputSize, int dim, float* gradInput
) {
    int o, i, d, mult, offset;
    i = 0;
    offset = 0;
    // break out small cases--not looping over d is much faster for low dims
    if (dim == 1) {
        for (o = 0; o < outputSize; o++) {
            for (; i < offset + len[o]; i++)
                gradInput[i] = gradOutput[o];
            offset = i;
        }
    } else if (dim == 2) {
        for (o = 0; o < outputSize; o++) {
            for (; i < offset + len[o] * 2; i += 2) {
                mult = o * 2;
                gradInput[i] = gradOutput[mult];
                gradInput[i + 1] = gradOutput[mult + 1];
            }
            offset = i;
        }
    } else {
        for (o = 0; o < outputSize; o++) {
            for (; i < offset + len[o] * dim; i += dim) {
                mult = o * dim;
                gradInput[i] = gradOutput[mult];
                for (d = 1; d < dim; d++)
                    gradInput[i + d] = gradOutput[mult + d];
            }
            offset = i;
        }
    }
}
