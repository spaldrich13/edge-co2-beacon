
#pragma once

static const int N_CHANNELS = 6;

static const float NORM_MU[6] = {
    -7.79250193f,  // [0] ax
    2.07272148f,  // [1] ay
    4.27278090f,  // [2] az
    0.41056770f,  // [3] gx
    -0.15372396f,  // [4] gy
    0.16609201f  // [5] gz
};

static const float NORM_SIGMA[6] = {
    1.59938383f,  // [0] ax
    1.88753140f,  // [1] ay
    3.04334497f,  // [2] az
    18.63519859f,  // [3] gx
    9.55679989f,  // [4] gy
    8.78757954f  // [5] gz
};