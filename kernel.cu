#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include<algorithm>
#include <math.h>
#include<time.h>
#define N 1000000
const int maxn=N*2;
#define mod 10000007
int a[maxn];
__global__ void merge_sort1(int *a, int *tmp, int size, int totNum)
{
	int i = blockIdx.x;
	int begin = i * size * 2;
	for(int i=0;begin<totNum;i++)
	{
			int l1 = begin;
			int l2 = begin + size;
			int end = begin + 2 * size;
			int s = begin;
			if (totNum < end)
				end = totNum;
			while (l1 < (begin + size) && l2 < end)
			{
				if (a[l1] <= a[l2])
					tmp[s++] = a[l1++];
				else
					tmp[s++] = a[l2++];
			}
			while (l1 < (begin + size))
				tmp[s++] = a[l1++];
			while (l2 < end)
				tmp[s++] = a[l2++];
			begin += gridDim.x * 2 * size;
	}
}
__global__ void merge_sort2(int *a, int *tmp, int size, int totNum)
{
	int offset = blockDim.x;
	int i = blockIdx.x;
	int j = threadIdx.x;
	int begin = i * size * 2;
	while (j < size * 2)
	{
		if (begin + size < totNum&&j < size * 2)
		{
			int l1 = begin;
			int l2 = begin + size;
			int end = begin + 2 * size;
			int s = begin;
			if (totNum < end)
				end = totNum;
			int r1 = begin + size;
			int r2 = end;
			if (j < end - begin)
			{
				int id = j + begin;
				if (id < r1)
				{
					if (a[id] <= a[l2])
					{
						int insertIndex = id;
						tmp[insertIndex] = a[id];
					}
					else
						if (a[id] > a[r2 - 1])
						{
							int insertIndex = id + r2 - l2;
							tmp[insertIndex] = a[id];
						}
						else
						{
							int l = l2 - 1, r = r2 - 1;
							while (r - l > 1)
							{
								int mid;
								if ((r + l) % 2 == 0)
									mid = (r + l) / 2;
								else
									mid = (r + l - 1) / 2;
								if (a[mid] <= a[id])
									r = mid;
								else
									l = mid;
							}
							int insertIndex = id + r;
							tmp[insertIndex] = a[id];
						}
				}
				else
				{
					if (a[id] < a[l1])
					{
						int insertIndex = id - (r1 - l1);
						tmp[insertIndex] = a[id];
					}
					else
						if (a[id] >= a[r1 - 1])
						{
							int insertIndex = id;
							tmp[insertIndex] = a[id];
						}
						else
						{
							int l = l1, r = r1;
							while (r - l > 1)
							{
								int mid;
								if ((r + l) % 2 == 0)
									mid = (r + l) / 2;
								else
									mid = (r + l - 1) / 2;
								if (a[mid] >= a[id])
									l = mid;
								else
									r = mid;
							}
							int insertIndex = id - r1 + l;
							tmp[insertIndex] = a[id];
						}
				}
			}
		}
		j += blockDim.x;
	}
}
cudaError_t solveWithCuda(int *a, int size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	printf("test1\n");
	int *dev_a = 0;
	int *dev_tmp = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_tmp, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int block = min(65500,size / 2);
	int numPerBlock = 1;
	int threadsPerBlock;
	int m1 = 1024;
	float calcTime;
	cudaEvent_t calcTimeS, calcTimeE;
	cudaEventCreate(&calcTimeS);
	cudaEventRecord(calcTimeS, 0);
	while (block)
	{
		threadsPerBlock = min(1024, numPerBlock);
		if (block >= m1)
			merge_sort1 << <block, 1 >> > (dev_a, dev_tmp, numPerBlock, size);
		else
			merge_sort2 << <block, threadsPerBlock >> > (dev_a, dev_tmp, numPerBlock, size);
		cudaMemcpy(dev_a, dev_tmp, size * sizeof(int), cudaMemcpyDeviceToDevice);
		if (block > 1)
			block = (block + 1) / 2;
		else
			block /= 2;
		numPerBlock *= 2;
	}
	cudaEventCreate(&calcTimeE);
	cudaEventRecord(calcTimeE, 0);
	cudaEventSynchronize(calcTimeE);
	cudaEventElapsedTime(&calcTime, calcTimeS, calcTimeE);
	printf("the first calc time used : %lf ms \n", calcTime);
	printf("test2\n");
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	printf("test3\n");
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate : %3.lf ms \n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


Error:
	cudaFree(dev_a);
	cudaFree(dev_tmp);
	return cudaStatus;
}
int main()
{
	srand(time(0));
	int n = N;
	for (int i = 0; i < n; i++)
		a[i] = rand() % mod;
	solveWithCuda(a, n);
	return 0;
}