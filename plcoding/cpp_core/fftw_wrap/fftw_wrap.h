#pragma once

#include <algorithm>
#include <complex>
#include <span>

#include <fftw3.h>

namespace fftw {
    struct fftw_deleter {
        void operator()(void* ptr) const { fftw_free(ptr); }
    };

    class real_array {
    public:
        explicit real_array(size_t size):
            data_(fftw_alloc_real(size)),
            span_(data_.get(), size) {
            if(!data_) { throw std::bad_alloc(); }
        }

        real_array(const real_array&) = delete;
        real_array& operator=(const real_array&) = delete;
        real_array(real_array&&) = delete;
        real_array& operator=(real_array&&) = delete;
        ~real_array() = default;

        std::span<double> span() { return span_; }

        std::span<const double> span() const { return span_; }

        auto ptr() { return data_.get(); }

    private:
        std::unique_ptr<double, fftw_deleter> data_;
        std::span<double> span_;
    };

    class complex_array {
    public:
        explicit complex_array(size_t size):
            data_(fftw_alloc_complex(size)),
            fftw_span_(data_.get(), size),
            std_span_(
                reinterpret_cast<std::complex<double>*>(data_.get()), size
            ) {
            if(!data_) { throw std::bad_alloc(); }
        }

        complex_array(const complex_array&) = delete;
        complex_array& operator=(const complex_array&) = delete;
        complex_array(complex_array&&) = delete;
        complex_array& operator=(complex_array&&) = delete;
        ~complex_array() = default;

        std::span<fftw_complex> fftw_span() { return fftw_span_; }

        std::span<const fftw_complex> fftw_span() const { return fftw_span_; }

        std::span<std::complex<double>> std_span() { return std_span_; }

        std::span<const std::complex<double>> std_span() const {
            return std_span_;
        }

        auto ptr() { return data_.get(); }

    private:
        std::unique_ptr<fftw_complex, fftw_deleter> data_;
        std::span<fftw_complex> fftw_span_;
        std::span<std::complex<double>> std_span_;
    };

    class wrapper {
    private:
        std::size_t q;
        fftw::real_array a_time, b_time, c_time;
        fftw::complex_array a_freq, b_freq, c_freq;
        fftw_plan a_plan, b_plan, c_plan;

    public:
        // initialization with sequence length
        explicit wrapper(std::size_t seq_len):
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
                static_cast<std::int32_t>(seq_len), a_time.ptr(), a_freq.ptr(),
                FFTW_MEASURE
            )),
            b_plan(fftw_plan_dft_r2c_1d(
                static_cast<std::int32_t>(seq_len), b_time.ptr(), b_freq.ptr(),
                FFTW_MEASURE
            )),
            c_plan(fftw_plan_dft_c2r_1d(
                static_cast<std::int32_t>(seq_len), c_freq.ptr(), c_time.ptr(),
                FFTW_MEASURE
            )) {};

        wrapper(const wrapper&) = delete;
        wrapper& operator=(const wrapper&) = delete;
        wrapper(wrapper&&) = delete;
        wrapper& operator=(wrapper&&) = delete;

        ~wrapper() {
            fftw_destroy_plan(a_plan);
            fftw_destroy_plan(b_plan);
            fftw_destroy_plan(c_plan);
        };

        // fast circular convolution
        void circonv(
            std::span<const double> in1, std::span<const double> in2,
            std::span<double> out
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
        };
    };
} // namespace fftw
