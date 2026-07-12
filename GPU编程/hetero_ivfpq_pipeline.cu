#define main hetero_sync_reference_main
#include "hetero_ivfpq_sync.cu"
#undef main

struct PipelineSlot {
    float* h_queries=0; uint32_t* h_candidates=0; uint32_t* h_ids=0; float* h_scores=0;
    float* d_queries=0; uint32_t* d_candidates=0; uint32_t* d_ids=0; float* d_scores=0;
    cudaStream_t stream=0; cudaEvent_t kernel_begin=0,kernel_end=0;
    size_t begin=0,count=0; bool active=false;
};

static void allocate_slot(PipelineSlot& s,size_t batch,size_t d,size_t p,size_t k) {
    CUDA_CHECK(cudaHostAlloc(&s.h_queries,batch*d*sizeof(float),cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&s.h_candidates,batch*p*sizeof(uint32_t),cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&s.h_ids,batch*k*sizeof(uint32_t),cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&s.h_scores,batch*k*sizeof(float),cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&s.d_queries,batch*d*sizeof(float))); CUDA_CHECK(cudaMalloc(&s.d_candidates,batch*p*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s.d_ids,batch*k*sizeof(uint32_t))); CUDA_CHECK(cudaMalloc(&s.d_scores,batch*k*sizeof(float)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&s.kernel_begin)); CUDA_CHECK(cudaEventCreate(&s.kernel_end));
}
static void release_slot(PipelineSlot& s) {
    cudaEventDestroy(s.kernel_end);cudaEventDestroy(s.kernel_begin);cudaStreamDestroy(s.stream);
    cudaFree(s.d_scores);cudaFree(s.d_ids);cudaFree(s.d_candidates);cudaFree(s.d_queries);
    cudaFreeHost(s.h_scores);cudaFreeHost(s.h_ids);cudaFreeHost(s.h_candidates);cudaFreeHost(s.h_queries);
}
static void submit_slot(PipelineSlot& s,size_t d,size_t p,size_t k,size_t nb,const float* db) {
    CUDA_CHECK(cudaMemcpyAsync(s.d_queries,s.h_queries,s.count*d*sizeof(float),cudaMemcpyHostToDevice,s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_candidates,s.h_candidates,s.count*p*sizeof(uint32_t),cudaMemcpyHostToDevice,s.stream));
    CUDA_CHECK(cudaEventRecord(s.kernel_begin,s.stream));
    rerank_topk<<<(unsigned)s.count,256,p*sizeof(float),s.stream>>>(db,s.d_queries,s.d_candidates,nb,d,p,k,s.d_ids,s.d_scores);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(s.kernel_end,s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.h_ids,s.d_ids,s.count*k*sizeof(uint32_t),cudaMemcpyDeviceToHost,s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.h_scores,s.d_scores,s.count*k*sizeof(float),cudaMemcpyDeviceToHost,s.stream)); s.active=true;
}

int main(int argc,char** argv) try {
    std::string dir="data/smoke"; size_t qn=16,k=10,nlist=8,nprobe=4,m=4,ks=16,train=128,iters=2,opq=1,p=64,batch=64,cpu_threads=8;
    for(int i=1;i<argc;++i) { std::string a=argv[i]; if(i+1>=argc)throw std::runtime_error("missing value for "+a);std::string v=argv[++i];
        if(a=="--data-dir")dir=v;else if(a=="--queries")qn=std::stoull(v);else if(a=="--k")k=std::stoull(v);else if(a=="--nlist")nlist=std::stoull(v);
        else if(a=="--nprobe")nprobe=std::stoull(v);else if(a=="--m")m=std::stoull(v);else if(a=="--ks")ks=std::stoull(v);
        else if(a=="--train-size")train=std::stoull(v);else if(a=="--kmeans-iters")iters=std::stoull(v);else if(a=="--opq-iters")opq=std::stoull(v);
        else if(a=="--rerank-p")p=std::stoull(v);else if(a=="--batch-size")batch=std::stoull(v);
        else if(a=="--cpu-threads")cpu_threads=std::stoull(v);else throw std::runtime_error("unknown option "+a);}
    std::string prefix=dir+((!dir.empty()&&(dir.back()=='/'||dir.back()=='\\'))?"":"/");size_t nb=0,d=0,nquery=0,qd=0;
    auto base=load_data<float>(prefix+"DEEP100K.base.100k.fbin",nb,d);auto queries=load_data<float>(prefix+"DEEP100K.query.fbin",nquery,qd);
    if(d!=qd||d%m||batch==0||cpu_threads==0)throw std::runtime_error("invalid dimensions, batch size, or CPU thread count");qn=std::min(qn,nquery);p=std::min(p,nb);k=std::min(k,p);batch=std::min(batch,qn);
    auto build0=std::chrono::steady_clock::now();IVFPQIndexSIMD index(base.data(),nb,d,nlist,m,ks,std::min(train,nb),iters,IVFPQInitMode::Uniform,opq);
    auto build1=std::chrono::steady_clock::now();float* db=0;CUDA_CHECK(cudaMalloc(&db,base.size()*sizeof(float)));CUDA_CHECK(cudaMemcpy(db,base.data(),base.size()*sizeof(float),cudaMemcpyHostToDevice));
    PipelineSlot slots[2];allocate_slot(slots[0],batch,d,p,k);allocate_slot(slots[1],batch,d,p,k);
    std::vector<uint32_t> all_candidates(qn*p,UINT32_MAX),gpu_ids(qn*k);std::vector<float> gpu_scores(qn*k);std::vector<size_t> counts(qn);
    double candidate_ms=0,kernel_ms=0;auto online0=std::chrono::steady_clock::now();size_t batch_id=0;
    for(size_t begin=0;begin<qn;begin+=batch,++batch_id) {
        PipelineSlot& s=slots[batch_id&1];
        if(s.active) {CUDA_CHECK(cudaStreamSynchronize(s.stream));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,s.kernel_begin,s.kernel_end));kernel_ms+=ms;
            std::copy(s.h_ids,s.h_ids+s.count*k,gpu_ids.begin()+s.begin*k);std::copy(s.h_scores,s.h_scores+s.count*k,gpu_scores.begin()+s.begin*k);s.active=false;}
        s.begin=begin;s.count=std::min(batch,qn-begin);auto c0=std::chrono::steady_clock::now();
        #pragma omp parallel for num_threads(cpu_threads) schedule(static)
        for(long long local_signed=0;local_signed<(long long)s.count;++local_signed) {size_t local=(size_t)local_signed;size_t q=begin+local;std::copy(queries.begin()+q*d,queries.begin()+(q+1)*d,s.h_queries+local*d);
            auto candidate=index.generate_candidates(queries.data()+q*d,nprobe,p);counts[q]=candidate.size();std::fill(s.h_candidates+local*p,s.h_candidates+(local+1)*p,UINT32_MAX);
            std::copy(candidate.begin(),candidate.end(),s.h_candidates+local*p);std::copy(candidate.begin(),candidate.end(),all_candidates.begin()+q*p);}
        candidate_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-c0).count();submit_slot(s,d,p,k,nb,db);
    }
    for(auto& s:slots)if(s.active){CUDA_CHECK(cudaStreamSynchronize(s.stream));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,s.kernel_begin,s.kernel_end));kernel_ms+=ms;
        std::copy(s.h_ids,s.h_ids+s.count*k,gpu_ids.begin()+s.begin*k);std::copy(s.h_scores,s.h_scores+s.count*k,gpu_scores.begin()+s.begin*k);s.active=false;}
    auto online1=std::chrono::steady_clock::now();size_t mismatch=0;float max_error=0;auto ref0=std::chrono::steady_clock::now();
    for(size_t q=0;q<qn;++q){auto ref=cpu_topk(base.data(),queries.data()+q*d,d,all_candidates.data()+q*p,counts[q],k);for(size_t r=0;r<ref.size();++r){
        max_error=std::max(max_error,std::fabs(ref[r].first-gpu_scores[q*k+r]));if(ref[r].second!=gpu_ids[q*k+r])++mismatch;}}
    auto ref1=std::chrono::steady_clock::now();release_slot(slots[1]);release_slot(slots[0]);cudaFree(db);
    double build_ms=std::chrono::duration<double,std::milli>(build1-build0).count(),online_ms=std::chrono::duration<double,std::milli>(online1-online0).count();
    double ref_ms=std::chrono::duration<double,std::milli>(ref1-ref0).count();
    std::cout<<"queries="<<qn<<" batch_size="<<batch<<" cpu_threads="<<cpu_threads<<" rerank_p="<<p<<" k="<<k<<"\n"
#if defined(__AVX2__) || defined(_M_AVX2)
      <<"cpu_simd=AVX2\n"
#else
      <<"cpu_simd=scalar_compat\n"
#endif
      <<"topk_id_mismatches="<<mismatch<<" max_score_abs_error="<<max_error<<"\n"
      <<(mismatch==0&&max_error<1e-4f?"CORRECTNESS PASS\n":"CORRECTNESS FAIL\n")<<"index_build_ms="<<build_ms<<"\n"<<"candidate_generation_work_ms="<<candidate_ms
      <<"\n"<<"gpu_kernel_work_ms="<<kernel_ms<<"\n"<<"pipeline_online_ms="<<online_ms<<"\n"<<"pipeline_us_per_query="<<online_ms*1000.0/qn
      <<"\n"<<"cpu_reference_rerank_ms="<<ref_ms<<"\n";return mismatch==0&&max_error<1e-4f?0:2;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
