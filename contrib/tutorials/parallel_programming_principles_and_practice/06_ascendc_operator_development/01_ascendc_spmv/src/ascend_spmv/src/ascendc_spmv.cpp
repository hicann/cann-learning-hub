#include "ascendc_spmv.hpp"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <acl/acl_rt_compile.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

#ifndef ASCENDC_SPMV_KERNEL_SOURCE
#error "ASCENDC_SPMV_KERNEL_SOURCE must point to the Ascend C kernel"
#endif

namespace spmv {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
bool check(aclError value, const char* call, std::string* error) {
    if (value == ACL_ERROR_NONE) return true;
    if (error) {
        *error = std::string(call) + " failed, aclError=" + std::to_string(value);
        if (const char* recent = aclGetRecentErrMsg()) *error += ": " + std::string(recent);
    }
    return false;
}
std::string read_kernel() {
    std::ifstream input(ASCENDC_SPMV_KERNEL_SOURCE, std::ios::binary);
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}
}  // namespace

struct AscendCSpmvBackend::Impl {
    aclrtStream stream = nullptr;
    aclrtcProg program = nullptr;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    aclrtArgsHandle arguments = nullptr;
    aclrtParamHandle parameter = nullptr;
    void* row_ptr = nullptr;
    void* col_idx = nullptr;
    void* values = nullptr;
    void* x = nullptr;
    void* y = nullptr;
    std::vector<char> elf;
    int64_t rows = 0;
    int32_t cols = 0;
    int32_t blocks = 32;
    bool acl_initialized = false;
    bool device_set = false;
};

template <typename ImplT>
bool initialize_runtime(ImplT* impl, int device, const std::string& source,
                        std::string* error) {
    if (source.empty()) {
        if (error) *error = "Ascend C kernel source is empty";
        return false;
    }
    if (!check(aclInit(nullptr), "aclInit", error)) return false;
    impl->acl_initialized = true;
    if (!check(aclrtSetDevice(device), "aclrtSetDevice", error)) return false;
    impl->device_set = true;
    return check(aclrtCreateStream(&impl->stream), "aclrtCreateStream", error);
}

template <typename ImplT>
bool compile_kernel(ImplT* impl, const std::string& source, std::string* error) {
    if (!check(aclrtcCreateProg(&impl->program, source.c_str(), "spmv_fp32",
                                0, nullptr, nullptr), "aclrtcCreateProg", error)) return false;
    const char* options[] = {"--npu-arch=dav-2201"};
    if (!check(aclrtcCompileProg(impl->program, 1, options),
               "aclrtcCompileProg", error)) return false;
    size_t elf_size = 0;
    if (!check(aclrtcGetBinDataSize(impl->program, &elf_size),
               "aclrtcGetBinDataSize", error)) return false;
    impl->elf.resize(elf_size);
    if (!check(aclrtcGetBinData(impl->program, impl->elf.data()),
               "aclrtcGetBinData", error)) return false;
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    aclrtBinaryLoadOptions load{};
    load.options = &option;
    load.numOpt = 1;
    if (!check(aclrtBinaryLoadFromData(impl->elf.data(), elf_size, &load,
                                       &impl->binary),
               "aclrtBinaryLoadFromData", error)) return false;
    return check(aclrtBinaryGetFunction(impl->binary, "spmv_fp32", &impl->function),
                 "aclrtBinaryGetFunction", error);
}

template <typename ImplT>
bool allocate_buffers(ImplT* impl, const CSRMatrix& matrix, std::string* error) {
    const size_t row_bytes = matrix.row_ptr.size() * sizeof(int32_t);
    const size_t col_bytes = matrix.col_idx.size() * sizeof(int32_t);
    const size_t value_bytes = matrix.values.size() * sizeof(float);
    const size_t x_bytes = static_cast<size_t>(matrix.cols) * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(matrix.rows) * sizeof(float);
    return check(aclrtMalloc(&impl->row_ptr, row_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(row_ptr)", error) &&
           check(aclrtMalloc(&impl->col_idx, col_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(col_idx)", error) &&
           check(aclrtMalloc(&impl->values, value_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(values)", error) &&
           check(aclrtMalloc(&impl->x, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(x)", error) &&
           check(aclrtMalloc(&impl->y, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(y)", error);
}

template <typename ImplT>
bool copy_matrix(ImplT* impl, const CSRMatrix& matrix, std::string* error) {
    const size_t row_bytes = matrix.row_ptr.size() * sizeof(int32_t);
    const size_t col_bytes = matrix.col_idx.size() * sizeof(int32_t);
    const size_t value_bytes = matrix.values.size() * sizeof(float);
    return check(aclrtMemcpy(impl->row_ptr, row_bytes, matrix.row_ptr.data(), row_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE), "H2D row_ptr", error) &&
           check(aclrtMemcpy(impl->col_idx, col_bytes, matrix.col_idx.data(), col_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE), "H2D col_idx", error) &&
           check(aclrtMemcpy(impl->values, value_bytes, matrix.values.data(), value_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE), "H2D values", error);
}

template <typename ImplT, typename ValueT>
bool append_argument(ImplT* impl, ValueT* value, const char* name, std::string* error) {
    return check(aclrtKernelArgsAppend(impl->arguments, reinterpret_cast<void**>(value),
                                       sizeof(ValueT), &impl->parameter), name, error);
}

template <typename ImplT>
bool bind_arguments(ImplT* impl, std::string* error) {
    if (!check(aclrtKernelArgsInit(impl->function, &impl->arguments),
               "aclrtKernelArgsInit", error)) return false;
    return append_argument(impl, &impl->row_ptr, "arg row_ptr", error) &&
           append_argument(impl, &impl->col_idx, "arg col_idx", error) &&
           append_argument(impl, &impl->values, "arg values", error) &&
           append_argument(impl, &impl->x, "arg x", error) &&
           append_argument(impl, &impl->y, "arg y", error) &&
           append_argument(impl, &impl->rows, "arg rows", error) &&
           check(aclrtKernelArgsFinalize(impl->arguments),
                 "aclrtKernelArgsFinalize", error);
}

AscendCSpmvBackend::AscendCSpmvBackend(int device) : impl_(new Impl), device_(device) {}
AscendCSpmvBackend::~AscendCSpmvBackend() {
    if (!impl_) return;
    if (impl_->binary) aclrtBinaryUnLoad(impl_->binary);
    if (impl_->y) aclrtFree(impl_->y);
    if (impl_->x) aclrtFree(impl_->x);
    if (impl_->values) aclrtFree(impl_->values);
    if (impl_->col_idx) aclrtFree(impl_->col_idx);
    if (impl_->row_ptr) aclrtFree(impl_->row_ptr);
    if (impl_->program) aclrtcDestroyProg(&impl_->program);
    if (impl_->stream) aclrtDestroyStream(impl_->stream);
    if (impl_->device_set) aclrtResetDevice(device_);
    if (impl_->acl_initialized) aclFinalize();
    delete impl_;
}

std::string AscendCSpmvBackend::name() const { return "Ascend C FP32 CSR SpMV (RTC)"; }

bool AscendCSpmvBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    if (!matrix.validate(error)) return false;
    const auto begin = Clock::now();
    const std::string source = read_kernel();
    if (!initialize_runtime(impl_, device_, source, error)) return false;
    if (!compile_kernel(impl_, source, error)) return false;
    impl_->rows = matrix.rows;
    impl_->cols = matrix.cols;
    if (!allocate_buffers(impl_, matrix, error)) return false;
    if (!copy_matrix(impl_, matrix, error)) return false;
    if (!bind_arguments(impl_, error)) return false;
    initialization_ms_ = ms(begin, Clock::now());
    return true;
}

bool AscendCSpmvBackend::run(const std::vector<float>& input, std::vector<float>* output,
                             BackendTimings* timings, std::string* error) {
    if (!output || input.size() != static_cast<size_t>(impl_->cols)) { if (error) *error = "input/output size mismatch"; return false; }
    const size_t x_bytes = input.size() * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(impl_->rows) * sizeof(float);
    const auto total_begin = Clock::now();
    const auto h2d_begin = total_begin;
    if (!check(aclrtMemcpy(impl_->x, x_bytes, input.data(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D x", error)) return false;
    const auto h2d_end = Clock::now();
    if (!check(aclrtLaunchKernelWithConfig(impl_->function, static_cast<uint32_t>(impl_->blocks), impl_->stream, nullptr, impl_->arguments, nullptr), "aclrtLaunchKernelWithConfig", error)) return false;
    if (!check(aclrtSynchronizeStream(impl_->stream), "aclrtSynchronizeStream", error)) return false;
    const auto kernel_end = Clock::now();
    output->resize(static_cast<size_t>(impl_->rows));
    if (!check(aclrtMemcpy(output->data(), y_bytes, impl_->y, y_bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H y", error)) return false;
    const auto end = Clock::now();
    if (timings) {
        timings->initialization_ms = initialization_ms_;
        timings->transfer_in_ms = ms(h2d_begin, h2d_end);
        timings->kernel_ms = ms(h2d_end, kernel_end);
        timings->transfer_out_ms = ms(kernel_end, end);
        timings->total_ms = ms(total_begin, end);
    }
    return true;
}
}  // namespace spmv
