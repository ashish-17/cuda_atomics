#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void copy(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

__global__ void copySharedMem(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM * TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

	//__syncthreads();

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}

__global__ void transposeNaive(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

__global__ void transposeCoalesced(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void postprocess(float *ref, float *res, int n)
{
	for (int i = 0; i < n; i++)
		if (res[i] != ref[i]) {
			printf("%d %f %f\n", i, res[i], ref[i]);
			printf("%25s\n", "*** FAILED ***");
			break;
		}
}

void runTransposeTest(int nIters) {
	int nx = 4096;
	int ny = 4096;

	int mem_size = nx*ny*sizeof(float);

	printf("\nStarting Test for %d x %d matrix with %d iterations\n", nx, ny, nIters);
	dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	float *h_idata = (float*)malloc(mem_size);
	float *h_cdata = (float*)malloc(mem_size);
	float *h_tdata = (float*)malloc(mem_size);
	float *gold    = (float*)malloc(mem_size);

	float *d_idata, *d_cdata, *d_tdata;
	cudaMalloc(&d_idata, mem_size);
	cudaMalloc(&d_cdata, mem_size);
	cudaMalloc(&d_tdata, mem_size);

	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			h_idata[j*nx + i] = j*nx + i;

	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			gold[j*nx + i] = h_idata[i*nx + j];

	cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
	cudaMemset(d_cdata, 0, mem_size);

	// Copy
	for (int i = 0; i < nIters; i++)
		copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);

	cudaDeviceSynchronize();

	cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost);

	postprocess(h_idata, h_cdata, nx*ny);

	// Copy shared
	cudaMemset(d_cdata, 0, mem_size);
	for (int i = 0; i < nIters; i++)
		copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);

	cudaDeviceSynchronize();

	cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost);

	postprocess(h_idata, h_cdata, nx*ny);

	// Transpose naive
	cudaMemset(d_tdata, 0, mem_size);
	for (int i = 0; i < nIters; i++)
		transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

	cudaDeviceSynchronize();
	cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
	postprocess(gold, h_tdata, nx * ny);


	// Transpose shared mem coaslesced
	cudaMemset(d_tdata, 0, mem_size);
	for (int i = 0; i < nIters; i++)
		transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

	cudaDeviceSynchronize();
	cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
	postprocess(gold, h_tdata, nx * ny);

	// Transpose shared mem without bank conflicts
	cudaMemset(d_tdata, 0, mem_size);
	for (int i = 0; i < nIters; i++)
		transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

	cudaDeviceSynchronize();
	cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
	postprocess(gold, h_tdata, nx * ny);

	cudaFree(d_tdata);
	cudaFree(d_cdata);
	cudaFree(d_idata);
	free(h_idata);
	free(h_tdata);
	free(h_cdata);
	free(gold);
	printf("\nFinishing Test\n");
}
