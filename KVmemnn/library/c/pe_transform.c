// Copyright 2004-present Facebook. All Rights Reserved.

void pe_transform(
    int d,
    int size,
    float* input,
    float* len,
    float* result
) {
    int j, k, s, add, offset;
    float frac, coeff;
    offset = 0;
    for (s = 0; s < size; s++) {
        for (j = 0; j < len[s]; j++) {
            frac = (float) (j + 1) / len[s];
            add = offset + j * d;
            for (k = 0; k < d; k++) {
                coeff = (1 - frac) - (float)(k + 1) / d * (1 - 2.0 * frac);
                result[add + k] = input[add + k] * coeff;
            }
        }
        offset += j * d;
    }
}
