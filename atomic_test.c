#define RESTRICTION_SIZE 32
#define BLOCK_SIZE 1024

__global__ void coalescedAtomicOnGlobalMem(int* data, int nElem)
{
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( data+i, 7);
	}
}

__global__ void addressRestrictedAtomicOnGlobalMem(int* data, int nElem)
{
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( data+(i&(RESTRICTION_SIZE-1)), 7);
	}
}

__global__ void warpRestrictedAtomicOnGlobalMem(int* data, int nElem)
{
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( data+(i>>5), 7);
	}
}

__global__ void sameAddressAtomicOnGlobalMem(int* data, int nElem)
{
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( data, 7);
	}
}

__global__ void coalescedAtomicOnSharedMem(int* data, int nElem)
{
	__shared__ int smem_data[BLOCK_SIZE];
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( smem_data+threadIdx.x, data[i]);
	}
}

__global__ void addressRestrictedAtomicOnSharedMem(int* data, int nElem)
{
	__shared__ int smem_data[BLOCK_SIZE];
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( smem_data+(threadIdx.x&(RESTRICTION_SIZE-1)), data[i&(RESTRICTION_SIZE-1)]);
	}
}

__global__ void warpRestrictedAtomicOnSharedMem(int* data, int nElem)
{
	__shared__ int smem_data[BLOCK_SIZE];
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( smem_data+(threadIdx.x>>5), data[i>>5]);

	}
}

__global__ void sameAddressAtomicOnSharedMem(int* data, int nElem)
{
	__shared__ int smem_data[BLOCK_SIZE];
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for ( unsigned int i = tid; i < nElem; i += blockDim.x*gridDim.x){
		atomicAdd( smem_data, data[0]);
	}
}

void runAtomicTest(int blockCount, int blockSize, int nIter) {
	printf("\nStarting test with Block count - %d and Block Size %d for %d iterations\n", blockCount, blockSize, nIter);

	int nElem = 2 << 24;
	int *data = (int*)malloc(sizeof(int) * nElem);
	int *d_data;
	cudaMalloc(&d_data, sizeof(int) * nElem);

	for (int i = 0; i < nElem; ++i) {
		data[i] = i%1024+1;
	}
	cudaMemcpy(d_data, data, sizeof(int) * nElem, cudaMemcpyHostToDevice);

	for (int i = 0; i < nIter; ++i) {
		coalescedAtomicOnGlobalMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		addressRestrictedAtomicOnGlobalMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		warpRestrictedAtomicOnGlobalMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		sameAddressAtomicOnGlobalMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		coalescedAtomicOnSharedMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		addressRestrictedAtomicOnSharedMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		warpRestrictedAtomicOnSharedMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < nIter; ++i) {
		sameAddressAtomicOnSharedMem<<<blockCount, blockSize>>>(d_data, nElem);
		cudaDeviceSynchronize();
	}

	free(data);
	cudaFree(d_data);
}
