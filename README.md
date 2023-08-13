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

#### Problem Decomposition:

1. **Dividing t Values:** Divide the t values into chunks, and assign each chunk to a different CPU core for Proximity Criteria checks.

2. **Data Distribution:** Distribute input points across the MPI processes. Each process will handle a subset of points.

3. **GPU-Accelerated Computation (CUDA):** Utilize GPU computation for efficient and parallel calculation of point coordinates (x, y) using CUDA kernels.

4. **Master Data Gathering and Combination (OpenMPI):** The master process gathers all computed data from worker processes, combines it, and sends the new combined data to all processes.

5. **Parallel Proximity Criteria Check (Parallel OpenMP):** Each worker process, after receiving the combined data, uses OpenMP to parallelize the Proximity Criteria check for each t value.

6. **Results Collection and Output (OpenMPI):** Collect and combine the results from all processes and generate the final output.

#### Parallelization Strategy:

1. **Master-Worker Architecture (OpenMPI):**
   - Rank 0 acts as the master, while other ranks act as workers.
   - The master reads input data and distributes work to worker processes.
   - Worker processes perform GPU-accelerated computation using CUDA and send results back to the master.

2. **GPU-Accelerated Computation (CUDA):**
   - Utilize the GPU to calculate point coordinates (x, y) efficiently using CUDA kernels.
   - Enable parallel processing of coordinate computations on the GPU for improved performance.

3. **Master Data Gathering and Combination (OpenMPI):**
   - The master receives computed data from worker processes.
   - Combines the data to create a new, combined dataset.
   - Distributes the combined dataset to all processes, including the master.

4. **Parallel Proximity Criteria Check (OpenMP):**
   - Each worker process, after receiving the combined data from the master, uses OpenMP to parallelize the Proximity Criteria check for each t value.
   - Ensure thread safety for accessing shared data structures during the parallel checks.

5. **Results Collection and Output (OpenMPI):**
   - All processes, including the master, send their computed results back to the master.
   - The master collects and combines the results from all processes.
   - The master then writes the final results to the output file.

#### Division of Responsibilities:

- **Master Process (Rank 0, OpenMPI + CUDA + OpenMP):**
  - Reads input data from the file.
  - Distributes work (points and t values) to worker processes.
  - Performs GPU-accelerated coordinate computation using CUDA.
  - Gathers and combines data from worker processes.
  - Distributes the combined dataset to all processes.
  - Collects computed results from all processes.
  - Writes final results to the output file.

- **Worker Processes (Rank 1 and beyond, OpenMPI + CUDA + OpenMP):**
  - Receive work (parameters and data) from the master process.
  - Perform GPU-accelerated computation using CUDA for coordinate calculations.
  - Send computed results back to the master process.
  - Receive the combined dataset from the master.
  - Perform a Parallel Proximity Criteria check using OpenMP on the combined data.

#### Summary:

By inserting a step for master data gathering and combination, the refined approach enhances data consolidation and distribution. The master process ensures accurate combination of worker results and facilitates efficient parallel Proximity Criteria checks using OpenMP. This approach maintains the Master-Worker architecture with MPI, utilizes GPU computation for coordinate calculations, and combines results using CUDA and OpenMP. The final results are collected, combined, and outputted in an effective and parallel manner.

![diagram-export-8_13_2023, 2_17_46 PM](https://github.com/DorFerenc/Parallel_Proximity_Criteria/assets/69848386/a85fcce5-71c3-4a70-af9e-dade56160320)

#IMPORTANT

* tCount needs to be dividable 4 or 2 DEPENDING on the number of proccess
