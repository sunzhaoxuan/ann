#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _MSC_VER
#define __builtin_prefetch(address, rw, locality) ((void)0)
#endif
#include "../ivf_pq_scan_simd.h"

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)

template<class T> std::vector<T> load_data(const std::string& path,size_t& n,size_t& d) {
    std::ifstream in(path.c_str(),std::ios::binary); if(!in) throw std::runtime_error("cannot open "+path);
    uint32_t nn=0,dd=0; in.read((char*)&nn,4); in.read((char*)&dd,4); n=nn; d=dd;
    std::vector<T> v(n*d); in.read((char*)v.data(),(std::streamsize)(v.size()*sizeof(T)));
    if(!in) throw std::runtime_error("truncated file "+path); return v;
}

__global__ void rerank_topk(const float* base,const float* queries,const uint32_t* candidates,
 size_t base_count,size_t dim,size_t p,size_t k,uint32_t* out_ids,float* out_scores) {
    const size_t q=blockIdx.x; extern __shared__ float scores[];
    for(size_t i=threadIdx.x;i<p;i+=blockDim.x) {
        const uint32_t id=candidates[q*p+i]; float s=-FLT_MAX;
        if(id<base_count) { s=0.0f; for(size_t j=0;j<dim;++j) s+=queries[q*dim+j]*base[(size_t)id*dim+j]; }
        scores[i]=s;
    }
    __syncthreads();
    if(threadIdx.x==0) for(size_t rank=0;rank<k;++rank) {
        size_t best=0;
        for(size_t i=1;i<p;++i) {
            const uint32_t a=candidates[q*p+i],b=candidates[q*p+best];
            if(scores[i]>scores[best] || (scores[i]==scores[best] && a<b)) best=i;
        }
        out_ids[q*k+rank]=candidates[q*p+best]; out_scores[q*k+rank]=scores[best]; scores[best]=-FLT_MAX;
    }
}

static std::vector<std::pair<float,uint32_t>> cpu_topk(const float* base,const float* query,size_t dim,
 const uint32_t* ids,size_t count,size_t k) {
    std::vector<std::pair<float,uint32_t>> v;
    for(size_t i=0;i<count;++i) { float s=0; for(size_t j=0;j<dim;++j) s+=query[j]*base[(size_t)ids[i]*dim+j]; v.push_back({s,ids[i]}); }
    std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){return a.first>b.first||(a.first==b.first&&a.second<b.second);});
    v.resize(std::min(k,v.size())); return v;
}

int main(int argc,char** argv) try {
    std::string dir="data/smoke"; size_t qn=16,k=10,nlist=8,nprobe=4,m=4,ks=16,train=128,iters=2,opq=1,p=64,batch_size=64;
    for(int i=1;i<argc;++i) { std::string a=argv[i]; if(i+1>=argc) throw std::runtime_error("missing value for "+a); std::string v=argv[++i];
        if(a=="--data-dir")dir=v; else if(a=="--queries")qn=std::stoull(v); else if(a=="--k")k=std::stoull(v);
        else if(a=="--nlist")nlist=std::stoull(v); else if(a=="--nprobe")nprobe=std::stoull(v); else if(a=="--m")m=std::stoull(v);
        else if(a=="--ks")ks=std::stoull(v); else if(a=="--train-size")train=std::stoull(v); else if(a=="--kmeans-iters")iters=std::stoull(v);
        else if(a=="--opq-iters")opq=std::stoull(v); else if(a=="--rerank-p")p=std::stoull(v);
        else if(a=="--batch-size")batch_size=std::stoull(v); else throw std::runtime_error("unknown option "+a); }
    const std::string prefix=dir+((!dir.empty()&&(dir.back()=='/'||dir.back()=='\\'))?"":"/"); size_t nb=0,d=0,nquery=0,qd=0;
    auto base=load_data<float>(prefix+"DEEP100K.base.100k.fbin",nb,d); auto queries=load_data<float>(prefix+"DEEP100K.query.fbin",nquery,qd);
    if(d!=qd||d%m||batch_size==0) throw std::runtime_error("invalid dimensions or batch size"); qn=std::min(qn,nquery); p=std::min(p,nb); k=std::min(k,p); batch_size=std::min(batch_size,qn);
    const auto total_begin=std::chrono::steady_clock::now();
    IVFPQIndexSIMD index(base.data(),nb,d,nlist,m,ks,std::min(train,nb),iters,IVFPQInitMode::Uniform,opq);
    const auto index_build_end=std::chrono::steady_clock::now();
    std::vector<uint32_t> candidates(qn*p,UINT32_MAX); std::vector<size_t> counts(qn);
    const auto candidate_begin=std::chrono::steady_clock::now();
    for(size_t q=0;q<qn;++q) { auto ids=index.generate_candidates(queries.data()+q*d,nprobe,p); counts[q]=ids.size(); std::copy(ids.begin(),ids.end(),candidates.begin()+q*p); }
    const auto candidate_end=std::chrono::steady_clock::now();
    float *db=0,*dq=0,*ds=0; uint32_t *dc=0,*di=0; CUDA_CHECK(cudaMalloc(&db,base.size()*4)); CUDA_CHECK(cudaMalloc(&dq,batch_size*d*4));
    CUDA_CHECK(cudaMalloc(&dc,batch_size*p*4)); CUDA_CHECK(cudaMalloc(&di,batch_size*k*4)); CUDA_CHECK(cudaMalloc(&ds,batch_size*k*4));
    CUDA_CHECK(cudaMemcpy(db,base.data(),base.size()*4,cudaMemcpyHostToDevice));
    std::vector<uint32_t> ids(qn*k); std::vector<float> scores(qn*k);
    double h2d_ms=0.0,d2h_ms=0.0,kernel_ms=0.0; cudaEvent_t ev0,ev1; CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    const auto gpu_phase_begin=std::chrono::steady_clock::now();
    for(size_t begin=0;begin<qn;begin+=batch_size) {
        const size_t current=std::min(batch_size,qn-begin); auto t0=std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(dq,queries.data()+begin*d,current*d*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dc,candidates.data()+begin*p,current*p*4,cudaMemcpyHostToDevice));
        auto t1=std::chrono::steady_clock::now(); h2d_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
        CUDA_CHECK(cudaEventRecord(ev0)); rerank_topk<<<(unsigned)current,256,p*4>>>(db,dq,dc,nb,d,p,k,di,ds); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1)); float elapsed=0; CUDA_CHECK(cudaEventElapsedTime(&elapsed,ev0,ev1)); kernel_ms+=elapsed;
        t0=std::chrono::steady_clock::now(); CUDA_CHECK(cudaMemcpy(ids.data()+begin*k,di,current*k*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scores.data()+begin*k,ds,current*k*4,cudaMemcpyDeviceToHost));
        t1=std::chrono::steady_clock::now(); d2h_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    const auto gpu_phase_end=std::chrono::steady_clock::now();
    size_t mismatch=0; float max_error=0;
    const auto reference_begin=std::chrono::steady_clock::now();
    for(size_t q=0;q<qn;++q) { auto ref=cpu_topk(base.data(),queries.data()+q*d,d,candidates.data()+q*p,counts[q],k); for(size_t r=0;r<ref.size();++r) {
        max_error=std::max(max_error,std::fabs(ref[r].first-scores[q*k+r])); if(ref[r].second!=ids[q*k+r]) {++mismatch; std::cerr<<"mismatch q="<<q<<" rank="<<r<<"\n";} } }
    const auto reference_end=std::chrono::steady_clock::now(); const auto total_end=reference_end;
    cudaEventDestroy(ev1);cudaEventDestroy(ev0);cudaFree(ds);cudaFree(di);cudaFree(dc);cudaFree(dq);cudaFree(db);
    const double candidate_ms=std::chrono::duration<double,std::milli>(candidate_end-candidate_begin).count();
    const double index_build_ms=std::chrono::duration<double,std::milli>(index_build_end-total_begin).count();
    const double gpu_phase_ms=std::chrono::duration<double,std::milli>(gpu_phase_end-gpu_phase_begin).count();
    const double reference_ms=std::chrono::duration<double,std::milli>(reference_end-reference_begin).count();
    const double total_ms=std::chrono::duration<double,std::milli>(total_end-total_begin).count();
    std::cout<<"queries="<<qn<<" batch_size="<<batch_size<<" rerank_p="<<p<<" k="<<k<<"\n"
      <<"topk_id_mismatches="<<mismatch<<" max_score_abs_error="<<max_error<<"\n"<<(mismatch==0&&max_error<1e-4f?"CORRECTNESS PASS\n":"CORRECTNESS FAIL\n");
    std::cout<<"index_build_ms="<<index_build_ms<<"\n"<<"candidate_generation_ms="<<candidate_ms<<"\n"<<"h2d_ms="<<h2d_ms<<"\n"<<"gpu_kernel_ms="<<kernel_ms<<"\n"
      <<"d2h_ms="<<d2h_ms<<"\n"<<"gpu_sync_phase_ms="<<gpu_phase_ms<<"\n"<<"cpu_reference_rerank_ms="<<reference_ms<<"\n"
      <<"online_hetero_ms="<<(candidate_ms+gpu_phase_ms)<<"\n"<<"validation_total_ms="<<total_ms<<"\n"
      <<"gpu_rerank_us_per_query="<<(gpu_phase_ms*1000.0/qn)<<"\n";
    return mismatch==0&&max_error<1e-4f?0:2;
} catch(const std::exception&e) {std::cerr<<"error: "<<e.what()<<"\n";return 1;}
