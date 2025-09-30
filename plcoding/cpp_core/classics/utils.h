#pragma once
#include "../fftw_wrap/fftw_wrap.h"
#include <cstddef>
#include <fftw3.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <span>

const uint32_t randa = 1103515245;
const uint32_t randc = 12345;

// get the inverse permutation
inline void get_inverse(std::size_t q, std::span<const std::size_t> randmap, std::span<std::size_t> lookups) {
    for (std::size_t i = 0; i < q; ++i) {
        lookups[randmap[i]] = i;
    }
}

// normalize the given vector
inline void normalize(std::size_t q, std::span<double> vector) {
    double tau = 0.0;
    for (std::size_t i = 0; i < q; ++i) {
        tau += vector[i];
    }
    for (std::size_t i = 0; i < q; ++i) {
        vector[i] /= tau;
    }
}

// generate a random permutation for Z_q
inline void gen_randmap(std::size_t q, std::span<std::size_t> randmap) {
    for (std::size_t i = 0; i < q; ++i) {
        randmap[i] = i;
    }
    uint32_t seed = q;
    for (std::size_t i = q - 1; i > 0; i--) {
        seed = seed * randa + randc;
        std::size_t j = seed % (i + 1);
        std::swap(randmap[i], randmap[j]);
    }
}

class FFTW3Wrapper {
private:
    std::size_t q;
    fftw::real_array a_time, b_time, c_time;
    fftw::complex_array a_freq, b_freq, c_freq;
    fftw_plan a_plan, b_plan, c_plan;
public:
    explicit FFTW3Wrapper(std::size_t seq_len);
    FFTW3Wrapper(const FFTW3Wrapper&) = delete;
    FFTW3Wrapper& operator=(const FFTW3Wrapper&) = delete;
    FFTW3Wrapper(FFTW3Wrapper&&) = delete;
    FFTW3Wrapper& operator=(FFTW3Wrapper&&) = delete;
    ~FFTW3Wrapper();
    
    void circonv(std::span<const double> in1, std::span<const double> in2, std::span<double> out);
};
