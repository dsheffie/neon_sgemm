#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <limits>
#include <type_traits>
#include <sys/time.h>
#include <time.h>
#include <arm_neon.h>

#include <Accelerate/Accelerate.h>
  
double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6*static_cast<double>(tv.tv_usec);
}


//template <typename T>

void gemm(float *Cg, float *Ag, float *Bg, int n, int m, int _k) {
  static const int VL = 4;
  static const int II_BLK = 4;
  static const int JJ_VL_BLK = 6;
  static const int JJ_BLK = VL * JJ_VL_BLK;

  static const int CACHE_BLK = 120;
  

  float32x4_t vC0_0, vC0_1, vC0_2, vC0_3, vC0_4,  vC0_5;
  float32x4_t vC1_0, vC1_1, vC1_2, vC1_3, vC1_4,  vC1_5;
  float32x4_t vC2_0, vC2_1, vC2_2, vC2_3, vC2_4,  vC2_5;
  float32x4_t vC3_0, vC3_1, vC3_2, vC3_3, vC3_4,  vC3_5;  

  float32x4_t vA, vB0, vB1, vB2, vB3, vB4, vB5;

  
  //the whole matrix
  for(int ii = 0; ii < n; ii += CACHE_BLK) {
    for(int jj = 0; jj < m; jj+=CACHE_BLK) {
      for(int kk = 0; kk < _k; kk += CACHE_BLK) {      
	for(int i = ii; i < (ii+CACHE_BLK); i+=II_BLK) {
	  for(int j = jj; j < (jj+CACHE_BLK); j+=JJ_BLK) {
	    
	    vC0_0 = vld1q_f32( &Cg[(i+0)*n+j+(0*VL)] );
	    vC0_1 = vld1q_f32( &Cg[(i+0)*n+j+(1*VL)] );
	    vC0_2 = vld1q_f32( &Cg[(i+0)*n+j+(2*VL)] );
	    vC0_3 = vld1q_f32( &Cg[(i+0)*n+j+(3*VL)] );
	    vC0_4 = vld1q_f32( &Cg[(i+0)*n+j+(4*VL)] );
	    vC0_5 = vld1q_f32( &Cg[(i+0)*n+j+(5*VL)] );                  
	    
	    vC1_0 = vld1q_f32( &Cg[(i+1)*n+j+(0*VL)] );
	    vC1_1 = vld1q_f32( &Cg[(i+1)*n+j+(1*VL)] );
	    vC1_2 = vld1q_f32( &Cg[(i+1)*n+j+(2*VL)] );
	    vC1_3 = vld1q_f32( &Cg[(i+1)*n+j+(3*VL)] );
	    vC1_4 = vld1q_f32( &Cg[(i+1)*n+j+(4*VL)] );
	    vC1_5 = vld1q_f32( &Cg[(i+1)*n+j+(5*VL)] );                  
	    
	    vC2_0 = vld1q_f32( &Cg[(i+2)*n+j+(0*VL)] );
	    vC2_1 = vld1q_f32( &Cg[(i+2)*n+j+(1*VL)] );
	    vC2_2 = vld1q_f32( &Cg[(i+2)*n+j+(2*VL)] );
	    vC2_3 = vld1q_f32( &Cg[(i+2)*n+j+(3*VL)] );
	    vC2_4 = vld1q_f32( &Cg[(i+2)*n+j+(4*VL)] );
	    vC2_5 = vld1q_f32( &Cg[(i+2)*n+j+(5*VL)] );                  
	    
	    vC3_0 = vld1q_f32( &Cg[(i+3)*n+j+(0*VL)] );
	    vC3_1 = vld1q_f32( &Cg[(i+3)*n+j+(1*VL)] );
	    vC3_2 = vld1q_f32( &Cg[(i+3)*n+j+(2*VL)] );
	    vC3_3 = vld1q_f32( &Cg[(i+3)*n+j+(3*VL)] );
	    vC3_4 = vld1q_f32( &Cg[(i+3)*n+j+(4*VL)] );
	    vC3_5 = vld1q_f32( &Cg[(i+3)*n+j+(5*VL)] );                  

	    for(int k = kk; k < (kk+CACHE_BLK); k++) {
	      vB0 = vld1q_f32(&Bg[(k)*m+j+(0*VL)]);
	      vB1 = vld1q_f32(&Bg[(k)*m+j+(1*VL)]);
	      vB2 = vld1q_f32(&Bg[(k)*m+j+(2*VL)]);
	      vB3 = vld1q_f32(&Bg[(k)*m+j+(3*VL)]);
	      vB4 = vld1q_f32(&Bg[(k)*m+j+(4*VL)]);
	      vB5 = vld1q_f32(&Bg[(k)*m+j+(5*VL)]);

	      //i = 0
	      vA = vdupq_n_f32(Ag[(i+0)*_k+k]);
	      vC0_0 = vmlaq_f32(vC0_0, vA, vB0);
	      vC0_1 = vmlaq_f32(vC0_1, vA, vB1);
	      vC0_2 = vmlaq_f32(vC0_2, vA, vB2);
	      vC0_3 = vmlaq_f32(vC0_3, vA, vB3);
	      vC0_4 = vmlaq_f32(vC0_4, vA, vB4);
	      vC0_5 = vmlaq_f32(vC0_5, vA, vB5);	  	  
	      
	      //i = 1
	      vA = vdupq_n_f32(Ag[(i+1)*_k+k]);
	      vC1_0 = vmlaq_f32(vC1_0, vA, vB0);
	      vC1_1 = vmlaq_f32(vC1_1, vA, vB1);
	      vC1_2 = vmlaq_f32(vC1_2, vA, vB2);
	      vC1_3 = vmlaq_f32(vC1_3, vA, vB3);
	      vC1_4 = vmlaq_f32(vC1_4, vA, vB4);
	      vC1_5 = vmlaq_f32(vC1_5, vA, vB5);	  	  
	      
	      //i = 2
	      vA = vdupq_n_f32(Ag[(i+2)*_k+k]);
	      vC2_0 = vmlaq_f32(vC2_0, vA, vB0);
	      vC2_1 = vmlaq_f32(vC2_1, vA, vB1);
	      vC2_2 = vmlaq_f32(vC2_2, vA, vB2);
	      vC2_3 = vmlaq_f32(vC2_3, vA, vB3);
	      vC2_4 = vmlaq_f32(vC2_4, vA, vB4);
	      vC2_5 = vmlaq_f32(vC2_5, vA, vB5);	  	  
	      
	      //i = 3
	      vA = vdupq_n_f32(Ag[(i+3)*_k+k]);
	      vC3_0 = vmlaq_f32(vC3_0, vA, vB0);
	      vC3_1 = vmlaq_f32(vC3_1, vA, vB1);
	      vC3_2 = vmlaq_f32(vC3_2, vA, vB2);
	      vC3_3 = vmlaq_f32(vC3_3, vA, vB3);
	      vC3_4 = vmlaq_f32(vC3_4, vA, vB4);
	      vC3_5 = vmlaq_f32(vC3_5, vA, vB5);	  	  
	    }
	    vst1q_f32( &Cg[(i+0)*n+j+(0*VL)], vC0_0 );
	    vst1q_f32( &Cg[(i+0)*n+j+(1*VL)], vC0_1 );
	    vst1q_f32( &Cg[(i+0)*n+j+(2*VL)], vC0_2 );
	    vst1q_f32( &Cg[(i+0)*n+j+(3*VL)], vC0_3 );
	    vst1q_f32( &Cg[(i+0)*n+j+(4*VL)], vC0_4 );
	    vst1q_f32( &Cg[(i+0)*n+j+(5*VL)], vC0_5 );    
	    
	    vst1q_f32( &Cg[(i+1)*n+j+(0*VL)], vC1_0 );
	    vst1q_f32( &Cg[(i+1)*n+j+(1*VL)], vC1_1 );
	    vst1q_f32( &Cg[(i+1)*n+j+(2*VL)], vC1_2 );
	    vst1q_f32( &Cg[(i+1)*n+j+(3*VL)], vC1_3 );
	    vst1q_f32( &Cg[(i+1)*n+j+(4*VL)], vC1_4 );
	    vst1q_f32( &Cg[(i+1)*n+j+(5*VL)], vC1_5 );    
	    
	    vst1q_f32( &Cg[(i+2)*n+j+(0*VL)], vC2_0 );
	    vst1q_f32( &Cg[(i+2)*n+j+(1*VL)], vC2_1 );
	    vst1q_f32( &Cg[(i+2)*n+j+(2*VL)], vC2_2 );
	    vst1q_f32( &Cg[(i+2)*n+j+(3*VL)], vC2_3 );
	    vst1q_f32( &Cg[(i+2)*n+j+(4*VL)], vC2_4 );
	    vst1q_f32( &Cg[(i+2)*n+j+(5*VL)], vC2_5 );    
	    
	    vst1q_f32( &Cg[(i+3)*n+j+(0*VL)], vC3_0 );
	    vst1q_f32( &Cg[(i+3)*n+j+(1*VL)], vC3_1 );
	    vst1q_f32( &Cg[(i+3)*n+j+(2*VL)], vC3_2 );
	    vst1q_f32( &Cg[(i+3)*n+j+(3*VL)], vC3_3 );
	    vst1q_f32( &Cg[(i+3)*n+j+(4*VL)], vC3_4 );
	    vst1q_f32( &Cg[(i+3)*n+j+(5*VL)], vC3_5 );    
	  }
	}
      }
    }
  }
}


template <typename T> 
void naive_gemm(T *Cg, T *Ag, T *Bg, int n, int m, int _k) {
  //the whole matrix
  for(int i = 0; i < n; i++) {
    for(int k = 0; k < _k; k++) {
      for(int j = 0; j < m; j++) {
	  Cg[i*n+j] += Ag[i*_k+k]*Bg[k*m+j];
      }
    }
  }

  
}


template <typename T>
T max(T a, T b) {
  return (a<b) ? b : a;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
T rand() {
  static const size_t M = static_cast<size_t>(1) << std::numeric_limits<T>::digits;
  T r = static_cast<T>((rand() & (M-1)));
  return (r / static_cast<T>(M/16));
}

#define print_var(X) std::cout << #X << " = " << X << "\n";

int main() {
  uint32_t l = 3000;
  assert((l % 12) == 0);
  uint32_t n = l ,m= l ,k= l;

  print_var(n);
  print_var(m);
  print_var(k);
    
  
  float *C0, *C1, *A, *B;

  
  C0 = new float[n*m];
  C1 = new float[n*m];
  A = new float[n*k];
  B = new float[k*m];

  memset(C0,0,sizeof(float)*n*m);
  memset(C1,0,sizeof(float)*n*m);

  uint64_t flops = 2UL * n * m * k;
  double gflops = static_cast<double>(flops)*1e-9;
  std::cout << "working set = " << (sizeof(float)*((n*m) + (n*k) + (k*m))) / 1024 << " kbytes\n";
  std::cout << "total gflops = " << gflops  << "\n";
  
  for(int i = 0; i < (n*k); i++) 
    A[i] = rand<float>();
  for(int i = 0; i < (m*k); i++) 
    B[i] = rand<float>();

  double t0 = timestamp();
  gemm(C0,A,B,n,m,k);
  double t1 = timestamp();

  std::cout << "optimized gemm " << gflops/(t1-t0) << " gflops/s\n";

  // while(1) {
  t0 = timestamp();
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,k,B,m,0.0f,C1,n);
  //naive_gemm<float>(C1,A,B,n,m,k);
  t1 = timestamp();
  std::cout << "naive gemm " << gflops/(t1-t0) << " gflops/s\n";
  //}

  
  float max_e = 0.0f;
  for(int i = 0; i < (n*m); i++) {
    float e = C0[i]-C1[i];
    e *= e;
    max_e = max(max_e, e);
    if(e > 1.0) {
      std::cout << "naive = " << C0[i] << "\n";
      std::cout << "opt   = " << C1[i] << "\n";
      break;
    }
  }
  std::cout << "Error = " << max_e << std::endl;

  delete [] (C0);
  delete [] (C1);
  delete [] (A);
  delete [] (B);
  return 0;
}
