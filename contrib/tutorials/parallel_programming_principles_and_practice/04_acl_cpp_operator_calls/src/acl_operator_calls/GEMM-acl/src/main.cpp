#include "acl_utils.hpp"

#include <aclnnop/aclnn_gemm.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
struct Options { int64_t m=1024, k=1024, n=1024; int warmup=10, repeat=100, device=0; };

Options parse(int argc, char** argv) {
    Options o;
    auto take = [&](int& i, const char* name) -> int64_t {
        if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
        return std::stoll(argv[++i]);
    };
    for (int i=1; i<argc; ++i) {
        std::string a=argv[i];
        if(a=="--m") o.m=take(i,"--m"); else if(a=="--k") o.k=take(i,"--k"); else if(a=="--n") o.n=take(i,"--n");
        else if(a=="--warmup") o.warmup=static_cast<int>(take(i,"--warmup")); else if(a=="--repeat") o.repeat=static_cast<int>(take(i,"--repeat"));
        else if(a=="--device") o.device=static_cast<int>(take(i,"--device")); else throw std::runtime_error("unknown argument: "+a);
    }
    if(o.m<=0||o.k<=0||o.n<=0||o.warmup<0||o.repeat<=0) throw std::runtime_error("dimensions must be positive, warmup >= 0 and repeat > 0");
    return o;
}

void cpu_gemm(const std::vector<float>& a,const std::vector<float>& b,std::vector<float>* c,int64_t m,int64_t k,int64_t n) {
    c->assign(static_cast<std::size_t>(m*n),0.0f);
    for(int64_t i=0;i<m;++i) for(int64_t p=0;p<k;++p) { const float av=a[static_cast<std::size_t>(i*k+p)]; for(int64_t j=0;j<n;++j) (*c)[static_cast<std::size_t>(i*n+j)] += av*b[static_cast<std::size_t>(p*n+j)]; }
}

double relative_error(const std::vector<float>& ref,const std::vector<float>& got) {
    long double diff=0, norm=0; for(std::size_t i=0;i<ref.size();++i){ const long double d=static_cast<long double>(got[i])-ref[i]; diff+=d*d; norm+=static_cast<long double>(ref[i])*ref[i]; }
    return std::sqrt(static_cast<double>(diff/std::max(norm,1e-30L)));
}

// Minimal RAII scope guard for the aclnn workspace: if aclnnGemm or the stream
// synchronize throws, the workspace is still released; the destructor never
// throws.
class Workspace {
public:
    explicit Workspace(std::size_t bytes) : ptr_(acl_demo::allocate_workspace(bytes)) {}
    ~Workspace() { acl_demo::free_workspace(ptr_); }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    void* get() const { return ptr_; }
private:
    void* ptr_ = nullptr;
};

void run_gemm(const Options& o,const std::vector<float>& a,const std::vector<float>& b,std::vector<float>* output) {
    acl_demo::Runtime runtime(o.device);
    acl_demo::DeviceBuffer da(a.size()*sizeof(float)), db(b.size()*sizeof(float)), dc(static_cast<std::size_t>(o.m*o.n)*sizeof(float)), dout(static_cast<std::size_t>(o.m*o.n)*sizeof(float));
    acl_demo::copy_to_device(da,a.data(),da.bytes()); acl_demo::copy_to_device(db,b.data(),db.bytes());
    std::vector<float> c_zero(static_cast<std::size_t>(o.m*o.n), 0.0f); acl_demo::copy_to_device(dc,c_zero.data(),dc.bytes());
    acl_demo::Tensor ta({o.m,o.k},ACL_FLOAT,da.data()), tb({o.k,o.n},ACL_FLOAT,db.data()), tc({o.m,o.n},ACL_FLOAT,dc.data()), tout({o.m,o.n},ACL_FLOAT,dout.data());

    const float alpha=1.0f, beta=0.0f; const int64_t trans_a=0, trans_b=0;
    // 0 = KEEP_DTYPE.  Use strict FP32 for this FP32 correctness experiment;
    // cubeMathType=1 permits FP32 down-precision and can exceed the 1e-6 target.
    const int8_t cube_math_type=0;
    uint64_t last_workspace_size=0;
    auto execute = [&]() {
        uint64_t workspace_size=0;
        aclOpExecutor* executor=nullptr;
        ACLNN_CHECK(aclnnGemmGetWorkspaceSize(ta.get(),tb.get(),tc.get(),alpha,beta,trans_a,trans_b,tout.get(),cube_math_type,&workspace_size,&executor));
        if (executor == nullptr) throw std::runtime_error("aclnnGemmGetWorkspaceSize returned a null executor");
        Workspace workspace(workspace_size);
        ACLNN_CHECK(aclnnGemm(workspace.get(),workspace_size,executor,runtime.stream()));
        ACL_CHECK(aclrtSynchronizeStream(runtime.stream()));
        last_workspace_size=workspace_size;
    };
    for(int i=0;i<o.warmup;++i) execute();
    double total_ms=0.0;
    for(int i=0;i<o.repeat;++i){ auto begin=std::chrono::steady_clock::now(); execute(); auto end=std::chrono::steady_clock::now(); total_ms+=std::chrono::duration<double,std::milli>(end-begin).count(); }
    output->resize(static_cast<std::size_t>(o.m*o.n)); acl_demo::copy_to_host(output->data(),output->size()*sizeof(float),dout);
    std::cout << "Actual Backend=ACL/CANN\nDevice ID=" << o.device
              << "\nWorkspace bytes=" << last_workspace_size
              << "\nTiming scope=aclnnGemm launch + stream synchronize"
              << "\nACL GEMM:\n  time = " << std::fixed << std::setprecision(6) << total_ms/o.repeat << " ms\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        const Options o=parse(argc,argv); std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f,1.0f);
        std::vector<float> a(static_cast<std::size_t>(o.m*o.k)), b(static_cast<std::size_t>(o.k*o.n)); for(float& v:a)v=dist(rng); for(float& v:b)v=dist(rng);
        std::vector<float> reference, result; cpu_gemm(a,b,&reference,o.m,o.k,o.n); run_gemm(o,a,b,&result); const double error=relative_error(reference,result);
        std::cout << "M=" << o.m << "\nK=" << o.k << "\nN=" << o.n << "\nCPU Reference Error=" << std::scientific << std::setprecision(9) << error << "\nCorrectness=" << (error < 1e-6 ? "PASS" : "FAIL") << "\n";
        return error < 1e-6 ? 0 : 1;
    } catch(const std::exception& e) { std::cerr << "GEMM ACL failed: " << e.what() << "\n"; return 2; }
}
