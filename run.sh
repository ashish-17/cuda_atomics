./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomic -blockNum 128 -blockSize 8
./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomicRow -blockNum 128 -blockSize 8
./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomicRandom -blockNum 128 -blockSize 8
./spmv -mat test_data/cant.mtx -ivec test_data/cant.vec -alg atomicTiled -blockNum 128 -blockSize 8
