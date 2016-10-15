#include <cstdio>
#include <cstdlib>
#include "my_cuda_get_time.cuh"
#include <cuda_profiler_api.h>

__global__ void kernel(int *a){
	int tx = blockDim.x*blockIdx.x+threadIdx.x;
	__shared__ int s_mem[64];
	s_mem[threadIdx.x]=a[tx];
	__syncthreads();
	a[tx]=s_mem[threadIdx.x]*2;
}

int main(int argc,char **argv){
	cudaProfilerStart();
	cudatimeStamp cutime(10);

	int gridx  = 128;
	int blockx = 64;
	int n = gridx*blockx;
	//host memory
	int *h=(int *)malloc(sizeof(int)*n);
	for(int i=0;i<n;i++) h[i]=i;
	//device memory
	int *d;
	cudaMalloc((void **)&d,sizeof(int)*n);

	//memcpy Host->Device
	cutime.stamp();
	cudaMemcpy(d,h,sizeof(int)*n,cudaMemcpyHostToDevice);

	//kernel
	cutime.stamp();
	kernel <<< gridx , blockx >>> (d);

	cutime.stamp();
	//memcpy Device->Host
	cudaMemcpy(h,d,sizeof(int)*n,cudaMemcpyDeviceToHost);
	cutime.stamp();

	cutime.print();
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));
	//memory free
	free(h);
	cudaFree(d);
	cudaProfilerStop();
	return 0;
}
