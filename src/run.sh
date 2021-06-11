set -v
g++ cpu.cpp -O3 -o cpu.out
nvcc gpu_base_1dim.cu -O3 -o gpu_base_1dim.out
nvcc gpu_base_2dim.cu -O3 -o gpu_base_2dim.out
nvcc gpu_base_log.cu -O3 -o gpu_base_log.out
nvcc gpu_fail.cu -O3 -o gpu_fail.out
nvcc gpu_shareMem.cu -O3 -o gpu_shareMem.out
nvcc gpu_shareMem_log.cu -O3 -o gpu_shareMem_log.out
g++ omp.cpp -fopenmp -O3 -o omp.out
./cpu.out 2000 2000
./gpu_base_1dim.out 2000 2000
./gpu_base_2dim.out 2000 2000
./gpu_base_log.out 2000 2000
./gpu_fail.out 2000 2000
./gpu_shareMem.out 2000 2000
./gpu_shareMem_log.out 2000 2000
./omp.out 2000 2000