#include <stdio.h>
#include <string.h>
#include "spmv.cuh"
#include "genresult.cuh"
#include "mmio.h"
#include "helper_cuda.h"

void logError(const char * errArg, const char * eMsg){
	if(eMsg != NULL)
		printf("Error: %s\n", eMsg);
	if(errArg != NULL)
		printf("Error found near: '%s'\n", errArg);
	puts("USAGE: spmv -mat [matrix file] -ivec [vector file] -alg [atomic|segment|design] -blockSize [blocksize] -blockNum [blocknum]");
	puts("Where the order of the parameters and string case do not matter");
	puts("Though the blockSize is optional (defaults to 1024)");
	puts("And the algorithms are:");
	puts("     - atomic =  simple atomics based approach");
	puts("     - segment = simple segment based scan approach");
	puts("     - design = design implementation");
}

typedef enum{
	CMDLN_ARG_NULL, 
	CMDLN_ARG_MAT = 1, 
	CMDLN_ARG_VEC = 2, 
	CMDLN_ARG_ALG = 4, 
	CMDLN_ARG_BLOCK = 8, 
	CMDLN_ARG_BLOCKNUM = 16, 
	CMDLN_ARG_ERR = 32, 
	CMDLN_ARG_DEVICEINFO, 
	CMDLN_ARG_ATOMIC_TEST,
	CMDLN_ARG_TRANSPOSE_TEST} CmdLnArg;

CmdLnArg getArgType(const char * argv){
	if(strcasecmp(argv, "-mat") == 0)
		return CMDLN_ARG_MAT;
	else if(strcasecmp(argv, "-ivec") == 0)
		return CMDLN_ARG_VEC;
	else if(strcasecmp(argv, "-alg") == 0)
		return CMDLN_ARG_ALG;
	else if(strcasecmp(argv, "-blockSize") == 0)
		return CMDLN_ARG_BLOCK;
	else if(strcasecmp(argv, "-blockNum") == 0)
		return CMDLN_ARG_BLOCKNUM;
	else if(strcasecmp(argv, "-deviceInfo") == 0)
		return CMDLN_ARG_DEVICEINFO;
	else if(strcasecmp(argv, "-atomicTest") == 0)
		return CMDLN_ARG_ATOMIC_TEST;
	else if(strcasecmp(argv, "-transposeTest") == 0)
		return CMDLN_ARG_TRANSPOSE_TEST;
	else
		return CMDLN_ARG_ERR;
}

typedef enum {ALG_ATOMIC, ALG_SEGMENT, ALG_DESIGN} AlgType;

int populateAlgType(const char * argv, AlgType * toPop){
	if(strcasecmp(argv, "atomic") == 0){
		*toPop = ALG_ATOMIC;
		return 1;
	}else if(strcasecmp(argv, "segment") == 0){
		*toPop = ALG_SEGMENT;
		return 1;
	}else if(strcasecmp(argv, "design") == 0){
		*toPop = ALG_DESIGN;
		return 1;
	}else return 0;
}

int doSpmv(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, AlgType how, int blockSize, int blockNum){
	switch(how){
		case ALG_ATOMIC:
			getMulAtomic(mat, vec, res, blockSize, blockNum);
			return 1;
		case ALG_SEGMENT:
			getMulScan(mat, vec, res, blockSize, blockNum);
			return 1;
		case ALG_DESIGN:
			getMulDesign(mat, vec, res, blockSize, blockNum);
			return 1;
		default:
			return 0;
	}
}

int verify(const int nz, const int M, const int *rIndex, const int *cIndex, const float *val, const float *vec, const float *res) {

	float *correct = (float*)malloc(sizeof(float) * M);
	memset(correct, 0, sizeof(float) * M);
	for (int i = 0; i < nz; ++i) {
		correct[rIndex[i]] += val[i] * vec[cIndex[i]];
	}

	int o = 0;

	for (int i = 0; i < M; ++i) {
		float l = correct[i] > 0 ? correct[i] : -1*correct[i];
		float m = res[i] > 0 ? res[i] : -1*res[i];
		float k = l - m > 0 ? l - m : m - l;
		float rel = k / l;
		if (rel > .01) {
			o++;
			printf("Yours - %f, correct - %f, Relative error - %f\n", res[i], correct[i], rel);
		}
	}

	return o;
}

void printDeviceInfo();

int main(int argc, char ** argv){
	/*if(argc != 11 && argc != 3){
	  logError(NULL, NULL);
	  return 1;
	  }*/

	//This is so that the arguments can be presented in any order with the blocksize defaulting to 1024
	int cumArgs = CMDLN_ARG_NULL;
	CmdLnArg argOrder[5];
	int i;
	for(i = 1; i < argc; i += 2){
		CmdLnArg currArg = getArgType(argv[i]);
		if(currArg == CMDLN_ARG_ERR || currArg & cumArgs){
			logError(argv[i], "Invalid or duplicate argument.");
			return 1;
		}else{
			argOrder[i/2] = currArg; //May the truncation be ever in our favor.
			cumArgs |= currArg;
		}
	}

	if(! (31 & cumArgs)){
		logError(NULL, "Missing arguments!");
		return 1;
	}

	char * mFile, * vFile;
	AlgType algo; //Si, debe ser algo!
	int blockSize = 128;
	int blockNum = 8;
	for(i = 0; i < (argc - 1)/2; i++){
		switch(argOrder[i]){
			case CMDLN_ARG_DEVICEINFO:
				printDeviceInfo();
				return 0;
			case CMDLN_ARG_ALG:
				if(!populateAlgType(argv[i * 2 + 2], &algo)){
					logError(argv[i * 2 + 2], "Unsupported algorithm");
					return 1;
				}
				break;
			case CMDLN_ARG_MAT:
				mFile = argv[i * 2 + 2];
				break;
			case CMDLN_ARG_VEC:
				vFile = argv[i * 2 + 2];
				break;
			case CMDLN_ARG_BLOCK:
				if(sscanf(argv[i * 2 + 2], "%d", &blockSize) != 1 || blockSize <= 0){
					logError(argv[i * 2 + 2], "Block size must be a positive integer (greater than 0)");
					return 1;
				}
				break;
			case CMDLN_ARG_BLOCKNUM:
				if(sscanf(argv[i * 2 + 2], "%d", &blockNum) != 1 || blockNum <= 0){
					logError(argv[i * 2 + 2], "Block num must be a positive integer (greater than 0)");
					return 1;
				}
				break;
			case CMDLN_ARG_ATOMIC_TEST:
				runAtomicTest(blockNum, blockSize, 5);
				return 0;
			case CMDLN_ARG_TRANSPOSE_TEST:
				runTransposeTest(100);
				return 0;

			default:
				puts("Logic is literally broken. This should never be seen!");
		}
	}

	printf("Reading matrix from %s\n", mFile);
	MatrixInfo * matrix = read_file(mFile);
	if(matrix == NULL){
		logError(mFile, "Error regarding matrix file.");
		return 1;
	}

	printf("Reading vector from %s\n", vFile);
	MatrixInfo * vector = read_vector_file(vFile, matrix->N);
	if(vector == NULL){
		logError(mFile, "Error regarding vector file.");
		return 1;
	}

	MatrixInfo * product = initMatrixResult(matrix->M, blockSize);
	cudaError_t err;
	if(!doSpmv(matrix, vector, product, algo, blockSize, blockNum)
			|| (err = cudaDeviceSynchronize()) != cudaSuccess
			|| !writeVect(product, "output.txt")){

		printf("\x1b[31m%s\x1b[0m\n", cudaGetErrorString(err));
		logError(NULL, "Failed to produce output");
	} else {
		printf("Verifying...\n");
		int o = verify(matrix->nz, matrix->M, matrix->rIndex, matrix->cIndex, matrix->val, vector->val, product->val);
		printf("%d Error rows \n", o);
	}


	freeMatrixInfo(matrix);
	freeMatrixInfo(vector);
	freeMatrixInfo(product);

	puts("So long and thank you for the fish!");
	return 0;
}

void printDeviceInfo() {

	printf("Starting...\n\n");
	printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;
	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		char msg[256];
		sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
				(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
		printf("%s", msg);

		printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
				deviceProp.multiProcessorCount,
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}
#else
		int memoryClock;
		cuDeviceGetAttribute(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
		int memBusWidth;
		cuDeviceGetAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
		int L2CacheSize;
		cuDeviceGetAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
		}
#endif
		printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
				deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
				deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
				deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
				deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
				deviceProp.maxThreadsDim[0],
				deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
		printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
				deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#ifdef WIN32
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}
}

