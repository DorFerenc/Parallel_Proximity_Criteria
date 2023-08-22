build:
# Compile main.c to main.o
	mpicxx -fopenmp -c main.c -o main.o
# Compile cFunctions.c to cFunctions.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
# Compile cudaFunctions.cu to cudaFunctions.o using nvcc
	nvcc -I./Common -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
# Compile point.c to point.o
	mpicxx -fopenmp -c point.c -o point.o
# Compile proximity_utils.c to proximity_utils.o
	mpicxx -fopenmp -c proximity_utils.c -o proximity_utils.o
# Link all object files to create the final executable mpiCudaOpemMP
	mpicxx -fopenmp -o mpiCudaOpemMP main.o cFunctions.o cudaFunctions.o point.o proximity_utils.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

clean:
# Remove object files and executable
	rm -f *.o ./mpiCudaOpemMP

run:
# Run the executable with 2 processes
	mpiexec -np 4 ./mpiCudaOpemMP

runOn2:
# Run the executable with 2 processes using a machinefile
	mpiexec -np 4 -machinefile ips.txt -map-by node ./mpiCudaOpemMP