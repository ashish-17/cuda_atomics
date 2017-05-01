# cuda_atomics
The atomic operations in the CUDA are crucial to some implementations but previous benchmarks reveal that, for an operation as simple as an addition of a single variable, performance of atomic operations can be thousands of times slower than a traditional read-modify-write cycle. 

In some previous projects - sparse matrix vector multiplication (spmv), we found that under the certain circumstances, cuda atomic instructions can also perform faster than non-atomic instructions. Besides that, other factors like sparsity pattern, storage schemes etc also may impact the performance of the implementation.

The goal of this project to to build an understanding of the performance of cuda atomic operations while varying other factors like storage schemes, sparsity pattern etc. and determine why the results are the way they are, and then repeating the cycle until we’re out of surprises. 

Proposed Solutions and Experiments -

We want to perform numerous experiments in varied settings to determine the cause of performance in results, starting off with simple experiment to gauge performance of atomicAdd() - 

Analyze the performance of a parallel counter - 
A counter is initialized on the host and the copied to GPU, where it is incremented using atomicAdd(), then the result is copied back to CPU.
Spread the atomicAdd() operations on multiple memory locations and analyze how it impacts the performance. Study the performance of atomic operations on continuous memory locations vs discontinuous memory locations. 
Coalesced atomic addition on global memory.
Atomic addition on a restricted address space in global memory.
Atomic addiction on same address in global memory.
Atomic addition, with all threads in warp operating on a single memory location.
Perform the above operations for shared memory and analyze its performance.
Perform the same operation but using Unified virtual addressing (cudamallochost()). Also compare the time for copying from host to device and device to host.
Compare it to performance of atomic operations  in case of CPU.

Then we plan to run the version of spmv using atomic versions with varying sparsity pattern, storage schemes and reading methodologies - 

Analyze the performance of a spmv (atomicAdd version) in varied settings - 
Sort the input entries in row order.
Input sorted in column order, which might have better performance because of less conflicts for atomicAdd.
Also try some obscure ordering methods like peano ordering, which claims to be more cache friendly and see if it can give better performance.
Tiling - divide the input in 2d tiles and partition the work among different threads and compare the performance with previous implementations.

Then we will try to improve the performance of above atomic oeprations by trying to aggregating the call to atomic operations. Following experiment will help analyze this -

Stream compaction (Filtering) -
	  Serial version:
for(uint i=0;i<length;i++)
    if(predicate(input[i])){
        output[j] = input[i];
        j++;
    }
}

Implement a parallel version of above code using cuda atomic operations.
Implement using warp aggregated atomics - 
Threads in the warp compute the total atomic increment for the warp and only one of them will use atomicAdd() to find the warp offset.
The warp offset is broadcasted to other threads, and each thread uses its own index with warp offset to find the correct position.
Segmented scan -
The segmented scan version of spmv in our previous implementation was significantly slower than the atomic version.
We will also try to analyze and improve this implementation of spmv.

Related Work

Following article presents a nice analysis of performance of atomic operations from a hardware point of view - 
http://www.strobe.cc/cuda_atomics/

This paper studies various storage layouts for sparse matrices and analyze the performance of sparse vector multiplication for each of them, but since the paper was written back in 2008 and performance of atomic operations has come far from that time, so I suspect the results might be old - 
"Efficient Sparse Matrix-Vector Multiplication on CUDA"
Nathan Bell and Michael Garland, in, "NVIDIA Technical Report NVR-2008-004", December 2008

Following paper also studies the performance of atomic operations in CUDA - 
“Massive Atomics for Massive Parallelism on GPUs”, Ian Egielski, Jesse Huang and Eddy Z. Zhang, ISMM’14

Following paper describes a memory layout, which is cache friendly for matrix multiplication, but the author tried this only for external memory optimization  - 
“Cache oblivious matrix multiplication using an element ordering based on the Peano curve” Michael Bader and Christoph Zenger 

