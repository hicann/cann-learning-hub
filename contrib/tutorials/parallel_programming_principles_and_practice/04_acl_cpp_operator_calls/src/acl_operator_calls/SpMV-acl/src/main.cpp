#include "acl_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
struct Options { int64_t rows=100000, cols=100000, nnz=1000000; int warmup=10, repeat=100, device=0; };
Options parse(int argc,char** argv){ Options o; auto next=[&](int& i,const char* n){if(i+1>=argc)throw std::runtime_error(std::string(n)+" needs a value");return std::stoll(argv[++i]);}; for(int i=1;i<argc;++i){std::string a=argv[i]; if(a=="--rows")o.rows=next(i,"--rows"); else if(a=="--cols")o.cols=next(i,"--cols"); else if(a=="--nnz")o.nnz=next(i,"--nnz"); else if(a=="--warmup")o.warmup=static_cast<int>(next(i,"--warmup")); else if(a=="--repeat")o.repeat=static_cast<int>(next(i,"--repeat")); else if(a=="--device")o.device=static_cast<int>(next(i,"--device")); else throw std::runtime_error("unknown argument: "+a);} if(o.rows<=0||o.cols<=0||o.nnz<0||o.nnz>o.rows*o.cols||o.warmup<0||o.repeat<=0)throw std::runtime_error("invalid dimensions or iteration count"); return o; }

void generate_csr(const Options& o,std::vector<int32_t>* row,std::vector<int32_t>* col,std::vector<float>* val,std::vector<float>* x){
    row->assign(static_cast<std::size_t>(o.rows+1),0); col->reserve(static_cast<std::size_t>(o.nnz)); val->reserve(static_cast<std::size_t>(o.nnz));
    std::mt19937 rng(42); std::uniform_int_distribution<int32_t> column(0,static_cast<int32_t>(o.cols-1)); std::uniform_real_distribution<float> value(-1.0f,1.0f);
    const int64_t base=o.nnz/o.rows, extra=o.nnz%o.rows; std::vector<int32_t> selected;
    for(int64_t r=0;r<o.rows;++r){ const int32_t count=static_cast<int32_t>(base+(r<extra)); selected.clear(); selected.reserve(count); while(static_cast<int32_t>(selected.size())<count){int32_t c=column(rng); if(std::find(selected.begin(),selected.end(),c)==selected.end())selected.push_back(c);} std::sort(selected.begin(),selected.end()); col->insert(col->end(),selected.begin(),selected.end()); for(int32_t i=0;i<count;++i)val->push_back(value(rng)); (*row)[static_cast<std::size_t>(r+1)]=static_cast<int32_t>(col->size()); }
    x->resize(static_cast<std::size_t>(o.cols)); for(float& v:*x)v=value(rng);
}
void cpu_spmv(const Options& o,const std::vector<int32_t>& row,const std::vector<int32_t>& col,const std::vector<float>& val,const std::vector<float>& x,std::vector<float>* y){y->assign(static_cast<std::size_t>(o.rows),0.0f);for(int64_t r=0;r<o.rows;++r)for(int32_t p=row[static_cast<std::size_t>(r)];p<row[static_cast<std::size_t>(r+1)];++p)(*y)[static_cast<std::size_t>(r)]+=val[static_cast<std::size_t>(p)]*x[static_cast<std::size_t>(col[static_cast<std::size_t>(p)])];}
double relative_error(const std::vector<float>& a,const std::vector<float>& b){long double d=0,n=0;for(std::size_t i=0;i<a.size();++i){long double z=static_cast<long double>(b[i])-a[i];d+=z*z;n+=static_cast<long double>(a[i])*a[i];}return std::sqrt(static_cast<double>(d/std::max(n,1e-30L)));}

// Minimal RAII scope guard for the ops-sparse descriptors: if any
// SPARSE_CHECK throws during setup, everything already created is still
// released; the destructor never throws. Destruction order is the exact
// reverse of creation (handle -> matrix -> vec_x -> vec_y):
// vec_y -> vec_x -> matrix -> handle.
struct SparseDescriptors {
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matrix = nullptr;
    aclsparseDnVecDescr_t vec_x = nullptr;
    aclsparseDnVecDescr_t vec_y = nullptr;
    ~SparseDescriptors() {
        if (vec_y) aclsparseDestroyDnVec(vec_y);
        if (vec_x) aclsparseDestroyDnVec(vec_x);
        if (matrix) aclsparseDestroySpMat(matrix);
        if (handle) aclsparseDestroy(handle);
    }
};
}

int main(int argc,char** argv){
    try {
        const Options o=parse(argc,argv); std::vector<int32_t> row,col; std::vector<float> val,x,reference,result; generate_csr(o,&row,&col,&val,&x); cpu_spmv(o,row,col,val,x,&reference);
        acl_demo::Runtime runtime(o.device); acl_demo::DeviceBuffer drow(row.size()*sizeof(int32_t)),dcol(col.size()*sizeof(int32_t)),dval(val.size()*sizeof(float)),dx(x.size()*sizeof(float)),dy(static_cast<std::size_t>(o.rows)*sizeof(float));
        acl_demo::copy_to_device(drow,row.data(),drow.bytes()); acl_demo::copy_to_device(dcol,col.data(),dcol.bytes()); acl_demo::copy_to_device(dval,val.data(),dval.bytes()); acl_demo::copy_to_device(dx,x.data(),dx.bytes()); std::vector<float> zero(static_cast<std::size_t>(o.rows),0.0f); acl_demo::copy_to_device(dy,zero.data(),dy.bytes());
        SparseDescriptors sparse;
        SPARSE_CHECK(aclsparseCreate(&sparse.handle)); SPARSE_CHECK(aclsparseSetStream(sparse.handle,runtime.stream()));
        SPARSE_CHECK(aclsparseCreateCsr(&sparse.matrix,o.rows,o.cols,o.nnz,drow.data(),dcol.data(),dval.data(),ACL_SPARSE_INDEX_32I,ACL_SPARSE_INDEX_32I,ACL_SPARSE_INDEX_BASE_ZERO,ACL_FLOAT));
        SPARSE_CHECK(aclsparseCreateDnVec(&sparse.vec_x,o.cols,dx.data(),ACL_FLOAT)); SPARSE_CHECK(aclsparseCreateDnVec(&sparse.vec_y,o.rows,dy.data(),ACL_FLOAT));
        const float alpha=1.0f,beta=0.0f; auto execute=[&](){SPARSE_CHECK(aclsparseSpMV(sparse.handle,ACL_SPARSE_OP_NON_TRANSPOSE,&alpha,sparse.matrix,sparse.vec_x,&beta,sparse.vec_y,ACL_FLOAT,ACL_SPARSE_SPMV_ALG_DEFAULT,nullptr)); ACL_CHECK(aclrtSynchronizeStream(runtime.stream()));};
        for(int i=0;i<o.warmup;++i)execute(); double total=0; for(int i=0;i<o.repeat;++i){auto b=std::chrono::steady_clock::now();execute();auto e=std::chrono::steady_clock::now();total+=std::chrono::duration<double,std::milli>(e-b).count();}
        result.resize(static_cast<std::size_t>(o.rows)); acl_demo::copy_to_host(result.data(),result.size()*sizeof(float),dy);
        const double error=relative_error(reference,result); std::cout<<"Actual Backend=ACL/CANN ops-sparse\nDevice ID="<<o.device<<"\nWorkspace bytes=0 (API default workspace)\nTiming scope=aclsparseSpMV launch + stream synchronize\nMatrix:\nrows = "<<o.rows<<"\ncols = "<<o.cols<<"\nnnz = "<<o.nnz<<"\n\nACL SpMV:\ntime = "<<std::fixed<<std::setprecision(6)<<total/o.repeat<<" ms\n\nCPU Reference Error="<<std::scientific<<std::setprecision(9)<<error<<"\nCorrectness="<<(error<1e-6?"PASS":"FAIL")<<"\n"; return error<1e-6?0:1;
    } catch(const std::exception& e){std::cerr<<"SpMV ACL failed: "<<e.what()<<"\n";return 2;}
}
