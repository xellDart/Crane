// FP8 (E4M3) vs bf16 GEMM microbenchmark on cuBLASLt — go/no-go spike for W8A8.
// Measures throughput + numerical error on the real MLP GEMM shapes of
// Vultron-8B and ops-4B, on this RTX 4090 (sm_89).
//
// Build: nvcc -O3 -arch=sm_89 fp8_bench.cu -lcublasLt -o fp8_bench
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CK(x) do{ cudaError_t _ck=(x); if(_ck!=cudaSuccess){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(_ck));exit(1);} }while(0)
#define LK(x) do{ cublasStatus_t _lk=(x); if(_lk!=CUBLAS_STATUS_SUCCESS){printf("cuBLASLt err %s @%d: %d\n",#x,__LINE__,(int)_lk);exit(1);} }while(0)

__global__ void f2bf(const float*x,__nv_bfloat16*y,long n){ long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) y[i]=__float2bfloat16(x[i]); }
__global__ void f2e4(const float*x,__nv_fp8_e4m3*y,float inv,long n){ long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) y[i]=__nv_fp8_e4m3(x[i]*inv); }
__global__ void bf2f(const __nv_bfloat16*x,float*y,long n){ long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) y[i]=__bfloat162float(x[i]); }

static cublasLtHandle_t lt;
static void* ws; static size_t wsSize=64ull*1024*1024;

// bf16 NN: C[M,N] = A[M,K]*B[K,N], all column-major.
float bench_bf16(int M,int N,int K,const __nv_bfloat16*A,const __nv_bfloat16*B,__nv_bfloat16*C,int iters){
  cublasLtMatmulDesc_t op; LK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t N_=CUBLAS_OP_N;
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N_,sizeof(N_)));
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N_,sizeof(N_)));
  cublasLtMatrixLayout_t Ad,Bd,Cd;
  LK(cublasLtMatrixLayoutCreate(&Ad,CUDA_R_16BF,M,K,M));
  LK(cublasLtMatrixLayoutCreate(&Bd,CUDA_R_16BF,K,N,K));
  LK(cublasLtMatrixLayoutCreate(&Cd,CUDA_R_16BF,M,N,M));
  cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
  LK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsSize,sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t heur{}; int ret=0;
  LK(cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Cd,pref,1,&heur,&ret));
  if(ret==0){printf("  bf16: no algo\n");return -1;}
  float alpha=1.f,beta=0.f;
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  for(int i=0;i<10;i++) LK(cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,C,Cd,C,Cd,&heur.algo,ws,wsSize,0));
  CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
  for(int i=0;i<iters;i++) LK(cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,C,Cd,C,Cd,&heur.algo,ws,wsSize,0));
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms=0; CK(cudaEventElapsedTime(&ms,s,e));
  cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd); cublasLtMatmulPreferenceDestroy(pref);
  return ms/iters;
}

// FP8 TN: D[M,N] = op(A)^T * B, A stored [K,M] col-major, B [K,N], D bf16.
float bench_fp8(int M,int N,int K,const __nv_fp8_e4m3*A,const __nv_fp8_e4m3*B,__nv_bfloat16*D,
                float*aScale,float*bScale,int iters,bool*ok){
  *ok=false;
  cublasLtMatmulDesc_t op; LK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t T_=CUBLAS_OP_T,N_=CUBLAS_OP_N;
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&T_,sizeof(T_)));
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N_,sizeof(N_)));
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&aScale,sizeof(aScale)));
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&bScale,sizeof(bScale)));
  int8_t fastAccum=1;
  LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fastAccum,sizeof(fastAccum)));
  cublasLtMatrixLayout_t Ad,Bd,Dd;
  LK(cublasLtMatrixLayoutCreate(&Ad,CUDA_R_8F_E4M3,K,M,K)); // [K,M]
  LK(cublasLtMatrixLayoutCreate(&Bd,CUDA_R_8F_E4M3,K,N,K)); // [K,N]
  LK(cublasLtMatrixLayoutCreate(&Dd,CUDA_R_16BF,M,N,M));    // [M,N]
  cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
  LK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsSize,sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t heur{}; int ret=0;
  cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Dd,Dd,pref,1,&heur,&ret);
  if(hs!=CUBLAS_STATUS_SUCCESS||ret==0){printf("  fp8: no algo (status %d ret %d)\n",(int)hs,ret);return -1;}
  float alpha=1.f,beta=0.f;
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  for(int i=0;i<10;i++){ auto st=cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,D,Dd,D,Dd,&heur.algo,ws,wsSize,0); if(st!=CUBLAS_STATUS_SUCCESS){printf("  fp8: matmul fail %d\n",(int)st);return -1;} }
  CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
  for(int i=0;i<iters;i++) cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,D,Dd,D,Dd,&heur.algo,ws,wsSize,0);
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms=0; CK(cudaEventElapsedTime(&ms,s,e));
  cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Dd); cublasLtMatmulPreferenceDestroy(pref);
  *ok=true; return ms/iters;
}

void run(const char*name,int M,int N,int K){
  long szA=(long)M*K, szB=(long)K*N, szC=(long)M*N;
  std::vector<float> hA(szA),hB(szB);
  unsigned seed=12345;
  auto rnd=[&](){ seed=seed*1664525u+1013904223u; return ((seed>>8)&0xffff)/65535.f*2.f-1.f; };
  for(long i=0;i<szA;i++) hA[i]=rnd()*0.1f;
  for(long i=0;i<szB;i++) hB[i]=rnd()*0.1f;
  float *dAf,*dBf; CK(cudaMalloc(&dAf,szA*4)); CK(cudaMalloc(&dBf,szB*4));
  CK(cudaMemcpy(dAf,hA.data(),szA*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBf,hB.data(),szB*4,cudaMemcpyHostToDevice));
  // bf16 buffers
  __nv_bfloat16 *Ab,*Bb,*Cb; CK(cudaMalloc(&Ab,szA*2)); CK(cudaMalloc(&Bb,szB*2)); CK(cudaMalloc(&Cb,szC*2));
  f2bf<<<(szA+255)/256,256>>>(dAf,Ab,szA); f2bf<<<(szB+255)/256,256>>>(dBf,Bb,szB);
  // fp8 buffers: A stored [K,M] = transpose of [M,K]. Our A is [M,K] col-major (lda=M).
  // For fp8 we need A as [K,M] col-major; simplest: treat the same data as B-like [K,N].
  // To keep it correct we quantize A into [K,M] layout by transposing on host.
  std::vector<float> hAt(szA);
  for(int r=0;r<M;r++) for(int c=0;c<K;c++) hAt[(long)c + (long)r*K] = hA[(long)r + (long)c*M]; // [K,M] col-major
  float* dAtf; CK(cudaMalloc(&dAtf,szA*4)); CK(cudaMemcpy(dAtf,hAt.data(),szA*4,cudaMemcpyHostToDevice));
  __nv_fp8_e4m3 *Af,*Bf; CK(cudaMalloc(&Af,szA)); CK(cudaMalloc(&Bf,szB));
  // per-tensor scale = absmax/448
  float amaxA=0,amaxB=0; for(long i=0;i<szA;i++) amaxA=fmaxf(amaxA,fabsf(hA[i])); for(long i=0;i<szB;i++) amaxB=fmaxf(amaxB,fabsf(hB[i]));
  float sA=amaxA/448.f, sB=amaxB/448.f;
  f2e4<<<(szA+255)/256,256>>>(dAtf,Af,1.f/sA,szA); f2e4<<<(szB+255)/256,256>>>(dBf,Bf,1.f/sB,szB);
  __nv_bfloat16* Df; CK(cudaMalloc(&Df,szC*2));
  float *dsA,*dsB; CK(cudaMalloc(&dsA,4)); CK(cudaMalloc(&dsB,4));
  CK(cudaMemcpy(dsA,&sA,4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsB,&sB,4,cudaMemcpyHostToDevice));

  int iters=50;
  float tb=bench_bf16(M,N,K,Ab,Bb,Cb,iters);
  bool ok; float tf=bench_fp8(M,N,K,Af,Bf,Df,dsA,dsB,iters,&ok);

  // numerical error: compare C(bf16) vs D(fp8), both -> f32
  double rel=-1;
  if(ok){
    std::vector<float> c(szC),d(szC); float *cf,*df; CK(cudaMalloc(&cf,szC*4)); CK(cudaMalloc(&df,szC*4));
    bf2f<<<(szC+255)/256,256>>>(Cb,cf,szC); bf2f<<<(szC+255)/256,256>>>(Df,df,szC);
    CK(cudaMemcpy(c.data(),cf,szC*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d.data(),df,szC*4,cudaMemcpyDeviceToHost));
    double num=0,den=0; for(long i=0;i<szC;i++){ double e=c[i]-d[i]; num+=e*e; den+=(double)c[i]*c[i]; }
    rel=sqrt(num/den); cudaFree(cf); cudaFree(df);
  }
  double gflop=2.0*M*N*K/1e9;
  printf("%-22s M=%d N=%d K=%d | bf16 %.3f ms (%.0f GFLOP/s) | fp8 ",name,M,N,K,tb,gflop/(tb/1e3));
  if(ok) printf("%.3f ms (%.0f GFLOP/s) | speedup %.2fx | rel-err %.4f\n",tf,gflop/(tf/1e3),tb/tf,rel);
  else printf("FAILED\n");
  cudaFree(dAf);cudaFree(dBf);cudaFree(Ab);cudaFree(Bb);cudaFree(Cb);cudaFree(dAtf);cudaFree(Af);cudaFree(Bf);cudaFree(Df);cudaFree(dsA);cudaFree(dsB);
}

int main(){
  LK(cublasLtCreate(&lt)); CK(cudaMalloc(&ws,wsSize));
  int M=1536; // padded token count (mult of 16, ~1481 real)
  printf("=== RTX 4090 (sm_89) cuBLASLt: FP8 E4M3 vs bf16, M=%d ===\n",M);
  run("Vultron mlp gate_up",M,24576,4096);
  run("Vultron mlp down",   M,4096,12288);
  run("ops mlp gate_up",    M,19456,2560);
  run("ops mlp down",       M,2560,9728);
  run("Vultron in_proj-ish",M,8192,4096);
  return 0;
}
