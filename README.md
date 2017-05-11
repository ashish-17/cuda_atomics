# cuda_atomics
The atomic operations in the CUDA are crucial to some implementations but previous benchmarks reveal that, for an operation as simple as an addition of a single variable, performance of atomic operations can be thousands of times slower than a traditional read-modify-write cycle. 

In some previous projects - sparse matrix vector multiplication (spmv), we found that under the certain circumstances, cuda atomic instructions can also perform faster than non-atomic instructions. Besides that, other factors like sparsity pattern, storage schemes etc also may impact the performance of the implementation.

The goal of this project to to build an understanding of the performance of cuda atomic operations while varying other factors like storage schemes, sparsity pattern etc. and determine why the results are the way they are, and then repeating the cycle until we’re out of surprises. 

Proposed Solutions and Experiments -

./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomic -blockNum 128 -blockSize 8
./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomicRow -blockNum 128 -blockSize 8
./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomicRandom -blockNum 128 -blockSize 8
./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomicTiled -blockNum 128 -blockSize 8

