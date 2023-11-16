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


double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6*static_cast<double>(tv.tv_usec);
}


//template <typename T>

void gemm(float *Cg, float *Ag, float *Bg, int n, int m, int _k) {
  static const int II_BLK = 3;
  static const int JJ_BLK = 8;
  static const int KK_BLK = 32;

  float32x4_t vC_tile[II_BLK][JJ_BLK];
  
  //the whole matrix
  for(int i = 0; i < n; i+=II_BLK) {
    for(int j = 0; j < m; j+=JJ_BLK) {

      for(int ii = 0; ii < II_BLK; ii++) {      
	for(int jj = 0; jj < (JJ_BLK/4); jj++) {
	  vC_tile[ii][jj] = vld1q_f32( &Cg[(i+ii)*n+j+(jj*4)] );
	}
      }


      for(int k = 0; k < _k; k+=KK_BLK) {	
	for(int ii = 0; ii < II_BLK; ii++) {
	  for(int kk = 0; kk < KK_BLK; kk++) {
	    float32x4_t vA = vdupq_n_f32(Ag[(i+ii)*_k+(k+kk)]);
	    for(int jj = 0; jj < (JJ_BLK/4); jj++) {
	      float32x4_t vB = vld1q_f32(&Bg[(k+kk)*m+j+(jj*4)]);
	      float32x4_t vT = vmulq_f32(vA, vB);	      
	      vC_tile[ii][jj] = vaddq_f32(vC_tile[ii][jj], vT);
	    }
	  }
	}
      }

      
      for(int ii = 0; ii < II_BLK; ii++) {      
	for(int jj = 0; jj < (JJ_BLK/4); jj++) {
	  vst1q_f32( &Cg[(i+ii)*n+j+(jj*4)], vC_tile[ii][jj]);
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
  uint32_t l = 240;
  assert((l % 6) == 0);
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

  t0 = timestamp();
  naive_gemm<float>(C1,A,B,n,m,k);
  t1 = timestamp();

  std::cout << "naive gemm " << gflops/(t1-t0) << " gflops/s\n";

  
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
