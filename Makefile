build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./Common -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -c point.c -o point.o
	mpicxx -fopenmp -c proximity_utils.c -o proximity_utils.o
	mpicxx -fopenmp -o mpiCudaOpemMP main.o cFunctions.o cudaFunctions.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile mf -map-by node ./mpiCudaOpemMP

