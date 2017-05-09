#include "genresult.cuh"
#include <sys/time.h>
#include <stdlib.h>

#define TILE_SIZE 256

extern void reorder_data(MatrixInfo* mat);
extern void mergeSortSeq(void* data, int item_size, int n, int (*comparator)(void*, void*));
extern void sort_matrix(MatrixInfo *mat, int (*comparator) (void*, void*));

__global__ void getMulAtomic_kernel(const int nz, const int *rIndex, const int *cIndex, const float *val, const float *vec, float *res) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int iter = nz % threadCount ? nz / threadCount + 1 : nz / threadCount;

	for (int i = 0; i < iter; ++i) {
		int dataId = threadId + i*threadCount;
		if (dataId < nz) {
			float data = val[dataId];
			int r = rIndex[dataId];
			int c = cIndex[dataId];
			float tmp = data * vec[c];
			atomicAdd(&res[r], tmp);
		}
	}
}

static inline int matComparatorRowOrder(void* a, void*b) {
	int aRowIdx = ((mat_t*)a)->rIndex;	
	int aColIdx = ((mat_t*)a)->cIndex;	
	int bRowIdx = ((mat_t*)b)->rIndex;	
	int bColIdx = ((mat_t*)b)->cIndex;	
	if (aRowIdx != bRowIdx) {
		return aRowIdx - bRowIdx;	
	} else {
		return aColIdx - bColIdx;
	}
}
static inline int matComparatorColOrder(void* a, void*b) {
	int aRowIdx = ((mat_t*)a)->rIndex;	
	int aColIdx = ((mat_t*)a)->cIndex;	
	int bRowIdx = ((mat_t*)b)->rIndex;	
	int bColIdx = ((mat_t*)b)->cIndex;	
	if (aColIdx == bColIdx) {
		return aRowIdx - bRowIdx;	
	} else {
		return aColIdx - bColIdx;
	}
}

static inline void random_reorder(MatrixInfo* mat) {
	printf("Randomizing  the matrix..\n");
	mat_t *mem = (mat_t*)malloc(sizeof(mat_t)*mat->nz);
	for (int i = 0; i < mat->nz; ++i) {
		mem[i].rIndex = mat->rIndex[i];
		mem[i].cIndex = mat->cIndex[i];
		mem[i].val = mat->val[i];
	}

	srand(time(NULL));

	for (int i = mat->nz-1; i > 0; --i) {
		int j = rand() % (i+1);

		mat_t tmp = mem[i];
		mem[i] = mem[j];
		mem[j] = tmp;
	}

	for (int i = 0; i < mat->nz; ++i) {
		mat->rIndex[i] = mem[i].rIndex;
		mat->cIndex[i] = mem[i].cIndex;
		mat->val[i] = mem[i].val;
	}

	free(mem);
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum, int mode){
	/*Allocate here...*/

	if (mode == 1) { // tilling
		printf("\nStart tiling for %d x %d matrix with %d nz\n", mat->M, mat->N, mat->nz);
		int* tmp_rIndex = (int*)malloc(sizeof(int) * mat->nz);
		int* tmp_cIndex = (int*)malloc(sizeof(int) * mat->nz);
		float* tmp_val = (float*)malloc(sizeof(float) * mat->nz);
		int count = 0;

		for (int x = TILE_SIZE; x < mat->M + TILE_SIZE; x += TILE_SIZE) {
			for (int y = TILE_SIZE; y < mat->N + TILE_SIZE; y += TILE_SIZE) { 
				for (int i = 0; i < mat->nz; ++i) {
					if (mat->cIndex[i] < y && mat->rIndex[i] < x && mat->cIndex[i] >= (y-TILE_SIZE) && mat->rIndex[i] >= (x - TILE_SIZE)) {
						tmp_rIndex[count] = mat->rIndex[i];
						tmp_cIndex[count] = mat->cIndex[i];
						tmp_val[count] = mat->val[i];
						count++;
					}
				}
			}
			printf("\n%d - %d", x, count);
		}

		memcpy(mat->cIndex, tmp_cIndex, sizeof(int)*mat->nz);
		memcpy(mat->rIndex, tmp_rIndex, sizeof(int)*mat->nz);
		memcpy(mat->val, tmp_val, sizeof(float)*mat->nz);
			printf("\n--------finish Tiling--------\n");
		free(tmp_rIndex);
		free(tmp_cIndex);
		free(tmp_val);
		if (count != mat->nz) {
			printf("\n--------Error Tiling--------\n");
		}

	} else if (mode == 2) { // Row sorted
		sort_matrix(mat, matComparatorRowOrder);
	} else if (mode == 3) { // Random
		random_reorder(mat);
	} else { // Col sorted
		sort_matrix(mat, matComparatorColOrder);
	}

	int *d_cIndex, *d_rIndex;
	float *d_val, *d_vec, *d_res;

	memset(res->val, 0, sizeof(float)*res->nz);
	cudaMalloc((void **)&d_cIndex, sizeof(int)*mat->nz);
	cudaMalloc((void **)&d_rIndex, sizeof(int)*mat->nz);
	cudaMalloc((void **)&d_val, sizeof(float)*mat->nz);
	cudaMalloc((void **)&d_vec, sizeof(float)*vec->nz);
	cudaMalloc((void **)&d_res, sizeof(float)*mat->M);

	cudaMemcpy(d_cIndex, mat->cIndex, mat->nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rIndex, mat->rIndex, mat->nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, mat->val, mat->nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, vec->val, vec->nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res->val, res->nz*sizeof(float), cudaMemcpyHostToDevice);

	/* Sample timing code */
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	/*Invoke kernels...*/
	getMulAtomic_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_res);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	/*Deallocate.*/
	cudaMemcpy(res->val, d_res, res->nz*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_cIndex);
	cudaFree(d_rIndex);
	cudaFree(d_val);
	cudaFree(d_vec);
	cudaFree(d_res);
}
