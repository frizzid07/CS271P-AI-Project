# CS271P-AI-Project
A repository to maintain AI project code for Group C505

## Group Members-
1. Tanmay Vikas Bhagwat (tvbhagwa@uci.edu)
2. Mandar Milind Bhalerao (mmbhaler@uci.edu)
2. Aditya Anil Bhave (aabhave@uci.edu)

## Submission Details-

The file structure is as follows:

1. run_tsp.py - This is the main file which contains all the driver functions to calculate the solution with SLS/BNB
2. tsp_bnb.py - This file contains all the code for Branch and bound
3. tsp_sls.py - This file contains all the code for stochastic local search
4. final_test_cases_and_results- This directory contains files with the input and output of the algorithms combined. For example, a file 5_50_15.out contains a randomly generated 5x5 symmetric matrix with a mean of 50 and standard deviation of 15. This file also contains the output metrics for both algorithms for given input matrix.

### Required Python packages for a successful execution-
NumPy, Collections, Random, Time, Math

In order to run the code, run the *run_tsp.py* file on command prompt. After running the file, you will first have to input number of locations, mean, and the standard deviation required to generate a distance matrix. We also create a new file where we write the distance matrix. This file will be in the directory named *final_test_cases_and_results*. Once the distance matrix is generated, you will see a menu with four options as below:

1. Branch and Bound DFS
2. Stochastic Local Search
3. Generate a New Distance Matrix
4. Stop the Execution

You can run both the algorithms on the same distance matrix that was generated by choosing one of the options above. The result of each algorithm would be written on the same file where the initial distance matrix was written. If you want to try the algorithms on a new distance matrix, you can choose option 3 which will redirect you to the first step of inputting a distance matrix. Choose option 4 to exit the program.
