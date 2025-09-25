#include "utils.h"

#include <algorithm>
#include <cstddef>
#include <span>

// initialization with sequence length
FFTW3Wrapper::FFTW3Wrapper(std::size_t seq_len):
    q(seq_len),
    // time domain sequences
    a_time(seq_len),
    b_time(seq_len),
    c_time(seq_len),
    // frequency domain sequences
    a_freq(seq_len / 2 + 1),
    b_freq(seq_len / 2 + 1),
    c_freq(seq_len / 2 + 1),
    // fftw3 plans
    a_plan(fftw_plan_dft_r2c_1d(
        static_cast<int>(seq_len), a_time.ptr(), a_freq.ptr(), FFTW_MEASURE
    )),
    b_plan(fftw_plan_dft_r2c_1d(
        static_cast<int>(seq_len), b_time.ptr(), b_freq.ptr(), FFTW_MEASURE
    )),
    c_plan(fftw_plan_dft_c2r_1d(
        static_cast<int>(seq_len), c_freq.ptr(), c_time.ptr(), FFTW_MEASURE
    )) {}

FFTW3Wrapper::~FFTW3Wrapper() {
    fftw_destroy_plan(a_plan);
    fftw_destroy_plan(b_plan);
    fftw_destroy_plan(c_plan);
}

// fast circular convolution
void FFTW3Wrapper::circonv(
    std::span<const double> in1, std::span<const double> in2, std::span<double> out
) {
    std::copy_n(in1.data(), this->q, this->a_time.ptr());
    std::copy_n(in2.data(), this->q, this->b_time.ptr());
    fftw_execute(a_plan);
    fftw_execute(b_plan);
    auto a_f = a_freq.std_span();
    auto b_f = b_freq.std_span();
    auto c_f = c_freq.std_span();
    std::ranges::transform(
        a_f, b_f, c_f.begin(),
        [](const auto& a, const auto& b) { return a * b; }
    );
    fftw_execute(c_plan);
    auto c_t = c_time.span();
    std::ranges::transform(c_t, out.begin(), [this](double val) {
        return val / static_cast<double>(this->q);
    });
}
