#include "utils.h"


// generate a random permutation for Z_q
void gen_randmap(int q, int* randmap) {
    for (int i = 0; i < q; ++i)
        randmap[i] = i;
    uint32_t seed = q;
    int temp;
    for (int i = q - 1; i > 0; i--) {
        seed = seed * randa + randc;
        int j = seed % (i + 1);
        temp = randmap[i];
        randmap[i] =randmap[j];
        randmap[j] = temp;
    }
}

// get the inverse permutation
void get_inverse(int q, int* randmap, int* lookups) {
    for (int i = 0; i < q; ++i)
        lookups[randmap[i]] = i;
}

// normalize the given vector
void normalize(int q, double* vector) {
    double tau = 0.0;
    for (int i = 0; i < q; ++i)
        tau += vector[i];
    for (int i = 0; i < q; ++i)
        vector[i] /= tau;
}

// initialization with sequence length
FFTW3Wrapper::FFTW3Wrapper(int seq_len) {
    this->q = seq_len;
    // time domain sequences
    a_time = fftw_alloc_real(seq_len);
    b_time = fftw_alloc_real(seq_len);
    c_time = fftw_alloc_real(seq_len);
    // frequency domain sequences
    a_freq = fftw_alloc_complex(seq_len / 2 + 1);
    b_freq = fftw_alloc_complex(seq_len / 2 + 1);
    c_freq = fftw_alloc_complex(seq_len / 2 + 1);
    // fftw3 plans
    a_plan = fftw_plan_dft_r2c_1d(seq_len, a_time, a_freq, FFTW_MEASURE);
    b_plan = fftw_plan_dft_r2c_1d(seq_len, b_time, b_freq, FFTW_MEASURE);
    c_plan  = fftw_plan_dft_c2r_1d(seq_len, c_freq, c_time, FFTW_MEASURE);
}

FFTW3Wrapper::~FFTW3Wrapper() {
    fftw_destroy_plan(a_plan);
    fftw_destroy_plan(b_plan);
    fftw_destroy_plan(c_plan);
    fftw_free(a_time);
    fftw_free(b_time);
    fftw_free(c_time);
    fftw_free(a_freq);
    fftw_free(b_freq);
    fftw_free(c_freq);
}

// fast circular convolution
void FFTW3Wrapper::circonv(const double* in1, const double* in2, double* out) {
    std::copy(in1, in1 + this->q, this->a_time);
    std::copy(in2, in2 + this->q, this->b_time);
    fftw_execute(a_plan);
    fftw_execute(b_plan);
    for (int i = 0; i < this->q / 2 + 1; ++i) {
        double a_re = a_freq[i][0], a_im = a_freq[i][1];
        double b_re = b_freq[i][0], b_im = b_freq[i][1];
        c_freq[i][0] = a_re * b_re - a_im * b_im;
        c_freq[i][1] = a_re * b_im + a_im * b_re;
    }
    fftw_execute(c_plan);
    for (int i = 0; i < this->q; ++i)
        out[i] = c_time[i] / this->q;
}
