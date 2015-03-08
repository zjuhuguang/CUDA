#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>



#include <string>
#include <sys/time.h> // for clock_gettime()
#include <unistd.h> // for usleep()
//	#define N 1008559420525856281
//#define N 0x6926C73F919FA3E7LL

//#define N 0xB8C8CBD2DAEE7DLL

__global__ void pollardKernel(int64_t m, int64_t num, int64_t* result_d, float* devData);
__global__ void cpyFunction(float* devData, int64_t* xd, int64_t m);


static double t0=0;
double Elapsed(void)
{
#ifdef _WIN32
    //  Windows version of wall time
    LARGE_INTEGER tv,freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&tv);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    double t = tv.QuadPart/(double)freq.QuadPart;
#else
    //  Unix/Linux/OSX version of wall time
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double t = tv.tv_sec+1e-6*tv.tv_usec;
#endif
    double s = t-t0;
    t0 = t;
    return s;
}
///////////////////////////////////////////////////////////////////////////////////////////////
// device version of f(x)
__device__ __forceinline__ int64_t fx_d(int64_t x, int64_t a, int64_t c) {
    return ( a * x * x + c);
}

// device version of binary gcd
__device__ int64_t gcd_d(int64_t u, int64_t v)
{
    int shift;
    
    /* GCD(0,v) == v; GCD(u,0) == u, GCD(0,0) == 0 */
    if (u == 0) return v;
    if (v == 0) return u;
    
    /* Let shift := lg K, where K is the greatest power of 2
     dividing both u and v. */
    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
        u >>= 1;
        v >>= 1;
    }
    
    while ((u & 1) == 0)
        u >>= 1;
    
    /* From here on, u is always odd. */
    do {
        /* remove all factors of 2 in v -- they are not common */
        /*   note: v is not zero, so while will terminate */
        while ((v & 1) == 0)  /* Loop X */
            v >>= 1;
        
        /* Now u and v are both odd. Swap if necessary so u <= v,
         then set v = v - u (which is even). For bignums, the
         swapping is just pointer movement, and the subtraction
         can be done in-place. */
        if (u > v) {
            int64_t t = v; v = u; u = t;}  // Swap u and v.
        v = v - u;                       // Here v >= u.
    } while (v != 0);
    
    /* restore common factors of 2 */
    return u << shift;
}
//////////////////////////////////////////////////////////////////////////////////////////////




__global__ void pollardKernel(int64_t m, int64_t num, int64_t* result_d, int64_t* xd)
{	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int64_t z, d = 1;
	int64_t x = xd[4 * threadId];
	int64_t y = xd[4 * threadId + 1];
	int64_t a = xd[4 * threadId + 2];
	int64_t c = xd[4 * threadId + 3];
	
    x = fx_d(x,a,c) % num;
    y = fx_d(fx_d(y,a,c),a,c) % num;
    z = abs(x-y);
    d = gcd_d(z,num);
    
    // copy updated state back into global memory
    xd[threadId * 4] = x;
    xd[threadId * 4 + 1] = y;

    // test to see if it found a factor
    if (d != 1 && d != num )
    {
        // if found, copy it into global syncronization variable "found"
        *result_d = d;
    }
 //   __syncthreads();
}


__global__ void cpyFunction(float* devData, int64_t* xd, int64_t m)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	xd[4 * threadId] = 0;
	xd[4 * threadId + 1] = 0;
	xd[4 * threadId + 2] = (int64_t)(devData[threadId * 2] * m);
	xd[4 * threadId + 3] = (int64_t)(devData[threadId * 2 + 1] * m);
}




int main(void)
{
	int64_t N =7576962498937463783;

    	int64_t m = sqrt(N);
    	int64_t result = 0;
  Elapsed();
//use curand to generate random parrallel
  size_t n =256 * 256;
  size_t i;
  curandGenerator_t gen;
  float *devData , *hostData;
  int64_t *xd;
  
  /* Allocate n floats on host */
  hostData = (float *) calloc(2 * n, sizeof(float));
  int64_t *x;
  /* Allocate n floats on device */
  cudaMalloc((void **) &devData, 2 * n * sizeof(float));
  x = (int64_t *) calloc(4 * n, sizeof(int64_t));
  cudaMalloc((void **) &xd, 4 * n*sizeof(int64_t));
  /* Create a Mersenne Twister pseudorandom number generator */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  /* Set seed */
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  
  /* Generate n floats on device */
  curandGenerateUniform(gen, devData, 2 * n);
  
  /* Copy device memory to host */
 cudaMemcpy(hostData , devData , 2 * n * sizeof(float), cudaMemcpyDeviceToHost);
/*
  for(i = 0; i < n; i++) {
    printf(" %1.4f\n", hostData[i]);
  }
*/
//allocate result
	int64_t* result_d;
    	cudaMalloc((void**)&result_d, sizeof(int64_t));
   	cudaMemcpy(result_d,&result, sizeof(int64_t), cudaMemcpyHostToDevice);
	cpyFunction<<<256, 256>>>(devData, xd, m);
	cudaThreadSynchronize();
	cudaMemcpy(x, xd, 4 * n * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaFree(devData);

 for(i = 0; i < 4 * n; i++) {
    printf(" %d\n", x[i]);
  }
	cudaThreadSynchronize();
//call the kernel
	do{
		pollardKernel<<<256, 256>>>(m, N, result_d, xd);
		cudaThreadSynchronize();
		cudaMemcpy(&result,result_d,sizeof(int64_t), cudaMemcpyDeviceToHost);
	  }
	while(result == 0);
	double Td = Elapsed();
	printf("%x\n%f", result, Td);
	cudaFree(result_d);
	cudaFree(xd);
//	free(result);

}

















