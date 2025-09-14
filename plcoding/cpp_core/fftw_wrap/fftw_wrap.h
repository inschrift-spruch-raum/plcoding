#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <memory>
#include <span>

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

        auto ptr() { return data_.get();}

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

        std::span<const std::complex<double>> std_span() const { return std_span_; }

        auto ptr() { return data_.get();}

    private:
        std::unique_ptr<fftw_complex, fftw_deleter> data_;
        std::span<fftw_complex> fftw_span_;
        std::span<std::complex<double>> std_span_;
    };
} // namespace fftw
