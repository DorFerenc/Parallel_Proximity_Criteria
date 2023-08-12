# Parallel_Proximity_Criteria

Parallel Implementation of Proximity Criteria
Final project
Course 10324, Parallel and Distributed Computation
2023 Fall Semester

A set of N points is placed in two-dimensional plane. Coordinates (x, y) of each point P are defined as follows:

x = ((x2 – x1) / 2 ) * sin (t*π /2) + (x2 + x1) / 2) 
y = a*x + b

where (x1, x2, a, b) are constant parameters predefined for each point P.

Problem Definition

We will say that some point P from the set satisfies a Proximity Criteria if there exist at least K points in the set with a distance from the point P less than a given value D.
Given a value of parameter t, we want to find if there exist at least 3 points that satisfies the Proximity Criteria 

Requirements

•	Perform checks for Proximity Criteria for tCount + 1 values of  t:
 t = 2 * i / tCount  - 1,          i = 0,  1,  2,  3, …,  tCount
		where tCount is a given integer number.
•	For each value of t find if there is three points that satisfy the Proximity Criteria. If such three points are found – don't continue evaluation for this specific value of t. 
•	The input file input.txt initially is known for one process only. The results must be written to the file output.txt by the same process. 
•	The computation time of the parallel program must be faster than sequential solution. 
•	Be ready to demonstrate your solution running on VLAB (two computers from different pools when using MPI)
•	No code sharing between students is allowed. Each part of code, if any, which was incorporated into your project must be referenced according to the academic rules.  
•	Be able to explain each line of the project code, including those that were reused from any source. 
•	The project that is not created properly (missing files, build or run errors) will not be accepted


Input data and Output Result of the project

The input file contains N in the first line - the number of point in the set, K – minimal number of points to satisfy the Proximity Criteria, distance D  and TCount. The next N lines contain parameters for every point in the set. One or more blanks are between the numbers in a file.
Input.txt
N   K   D   TCount
id   x1    x2    a    b
id   x1    x2    a    b
id   x1    x2    a    b
…
id   x1    x2    a    b

For example
4      2      1.23     100
0    2.2     1.2      2       45.07
1    -1       26.2    4,4    -3.3
2    -43.3   12.2   4.7     20
3    11.0    -6.6    12.5   23. 

Output.txt
The output file contains information about results found for points that satisfies the Proximity Criteria. 
•	For each t that 3 points satisfying the Proximity Criteria were found, it contains a line with the parameter t and ID of these 3 points
Points  pointID1, pointID2, pointID3 satisfy Proximity Criteria at t = t1 
Points  pointID4, pointID5, pointID6 satisfy Proximity Criteria at t = t2
Points  pointID7, pointID8, pointID9 satisfy Proximity Criteria at t = t3

•	In case that the points were not found for any t, the program outputs:
There were no 3 points found for any t.


____________________________________________________________________________________________________

It seems like you want to implement a master-worker pattern using MPI to distribute computation tasks among multiple processes, and then perform distance calculations using OpenMP within each worker. Here's an outline of how you can achieve this:

1. **Master Process (Rank 0):**

   - Read input data and distribute necessary parameters and data to worker processes.
   - Perform GPU-accelerated computation using CUDA.
   - Receive results from worker and his own processes after computing workerPointsTcount with GPU.
   - Collect and combine the results from all worker processes.
   - Build a new array containing all points from each worker for all t values.
   - Send the new FINAL_POINTS array to each worker.
   - Perform Parallel Proximity Criteria Check using OpenMP on the new array.
   - Collect and combine the results from all worker processes.
   - Write the combined results to the output file.

2. **Worker Processes (Rank 1 and beyond):**

   - Receive parameters and data from the master process.
   - Perform GPU-accelerated computation using CUDA.
   - Send the computed results to the MASTER.
   - Receive the new FINAL_POINTS array from the MASTER.
   - Perform Parallel Proximity Criteria Check using OpenMP on the new array.
   - Send computed results back to the master process.


#### Problem Decomposition:

1. **Dividing t Values:** Divide the t values into chunks, and assign each chunk to a different CPU core for Proximity Criteria checks.

2. **Data Distribution:** Distribute input points across the MPI processes. Each process will handle a subset of points.

3. **Coordinate Computation:** Use the GPU to accelerate the computation of coordinates (x, y) for each point.

4. **Proximity Criteria Check:** Parallelize the Proximity Criteria check for each t value across multiple CPU cores using OpenMP.

5. **Results Combination:** Collect and combine the results from all processes to generate the final output.

#### Parallelization Strategy:

1. **Master-Worker Architecture (OpenMPI):**
   - One computer acts as the master and the other as a worker.
   - The master reads input data and distributes work to worker processes.
   - Workers perform Proximity Criteria checks using OpenMP.
   - Workers send results back to the master.

2. **GPU-Accelerated Computation (CUDA):**
   - Use the Nvidia GPU for efficient computation of coordinates (x, y) for each point.
   - Design CUDA kernels to perform coordinate calculations in parallel on the GPU.

3. **Parallel Proximity Criteria Check (OpenMP):**
   - For each chunk of t values assigned to a CPU core, use OpenMP to parallelize the Proximity Criteria check.
   - Ensure thread safety for accessing shared data structures.

#### Potential Challenges and Solutions:

1. **Load Balancing:**
   - Uneven distribution of points or t values across CPU cores or processes.
   - Solution: Implement dynamic load balancing techniques to distribute work more evenly.

2. **Memory Management:**
   - Efficiently manage memory for input data, intermediate results, and GPU computations.
   - Solution: Properly allocate, transfer, and release memory on both CPU and GPU.

3. **Synchronization and Communication:**
   - Synchronizing threads and processes, and handling communication overhead between processes.
   - Solution: Use synchronization mechanisms (OpenMP's synchronization constructs, MPI communication routines) judiciously and optimize communication patterns.

4. **Data Consistency:**
   - Ensuring consistent data across CPU and GPU memory spaces.
   - Solution: Manage data transfers between CPU and GPU memory regions carefully, and ensure synchronization points.

5. **GPU Thread Hierarchy:**
   - Properly designing and launching CUDA kernels to utilize the GPU's thread hierarchy.
   - Solution: Understand and apply the CUDA thread hierarchy (blocks, threads, grids) effectively for optimal performance.

6. **Performance Bottlenecks:**
   - Identifying and addressing potential performance bottlenecks in the GPU-accelerated computation.
   - Solution: Profile and optimize CUDA kernels to maximize GPU utilization and minimize memory transfers.

7. **Fault Tolerance:**
   - Handling failures of processes or GPU-related issues.
   - Solution: Implement error handling and recovery mechanisms to ensure robustness.

8. **Data Integrity:**
   - Ensuring that combined results are accurate and complete.
   - Solution: Implement proper synchronization and data aggregation mechanisms in the master process.

#### Division of Responsibilities:

- **Master Process (OpenMPI):**
  - Reads input data from the file.
  - Distributes work (points and t values) to worker processes.
  - Collects and combines results from workers.
  - Writes final results to the output file.

- **Worker Processes (OpenMPI + OpenMP):**
  - Receive work from the master process.
  - Use OpenMP to parallelize the Proximity Criteria check for each t value.
  - Send computed results back to the master process.

- **GPU Computation (CUDA):**
  - Perform the computation of coordinates (x, y) for each point using CUDA kernels.

#### Summary:

This approach utilizes OpenMPI for inter-process communication, OpenMP for intra-process parallelization on CPU cores, and CUDA for GPU-accelerated computation. Dividing responsibilities among different components and addressing potential challenges will help you create an efficient and robust parallel solution for the "Parallel Implementation of Proximity Criteria" task in your computing environment.

![image](https://github.com/DorFerenc/Parallel_Proximity_Criteria/assets/69848386/86a24b33-2e4f-4bf1-85db-0966de04184f)
