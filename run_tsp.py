from collections import deque
import numpy as np
import math
import random
import time

import tsp_bnb
import tsp_sls

def write_distance_matrix(n, mean, sigma):
    distance_matrix = np.zeros((n, n))
    random_distance = []
    num_distance = int(n * (n-1) / 2)
    for _ in range(num_distance):
        distance = 0
        while distance <= 0:
            distance = np.random.normal(mean, sigma)
        random_distance.append(distance)
    iu = np.triu_indices(n, 1)
    distance_matrix[iu] = random_distance
    distance_matrix += distance_matrix.T
    distance_matrix = distance_matrix.astype(float)
    
    np.savetxt(
        f"final_test_cases_and_results/{n}_{mean}_{sigma}.out",
        distance_matrix,
        delimiter=" ",
        fmt="%1.4f",
        header=str(n),
        comments="",
    )

    return distance_matrix, f"final_test_cases_and_results/{n}_{mean}_{sigma}.out"

def write_result(file_path, choice, arg1, arg2):
    with open(file_path, "a") as f:
        if choice == 1:
            f.write("\n\nResults from Branch and Bound:")
            f.write("\n\nTime taken: " +str(arg2)+ " seconds")
            f.write("\nLower bound from Heuristic function: " +str(arg1.lower_bound))
            f.write("\nTotal number of nodes explored: " +str(arg1.nodes_explored))
            f.write("\n\nFinal path: ")
            output = arg1.path.copy()
            while output:
                f.write(str(output.popleft()) + " -> ")
            f.write(str(arg1.start.label))
            output.clear()
            f.write("\nCost of final path: " +str(arg1.upper_bound))
        elif choice == 2:
            f.write("\n\nResults from Stochastic Local Search:")
            f.write("\n\nTime taken: " +str(arg2[2])+ " seconds")
            f.write("\nFinal path: " +arg1)
            f.write("\nCost of final path: " +str(arg2[1]))

if __name__ == "__main__":
    while True:
        nodes = int(input("Enter the number of locations: "))
        if nodes > 1:
            mean = int(input("Enter the mean: "))
            sigma = int(input("Enter the standard deviation: "))
        else:
            print("Please enter valid number of locations\n")
            continue
        
        dist_matrix, file_path = write_distance_matrix(nodes, mean, sigma)

        while True:
            print("\nWhich algorithm would you like to run for TSP?\n1. Branch and Bound DFS\n2. Stochastic Local Search\n3. Generate a New Distance Matrix: \n4. Stop the Execution")
            choice = int(input("Enter your choice: "))
            if choice == 1:
                algorithm = tsp_bnb.Graph(nodes)
                algorithm.start.matrix = dist_matrix
                time_taken = algorithm.run()
                write_result(file_path, choice, algorithm, time_taken)
            elif choice == 2:
                algorithm = tsp_sls.SLS(dist_matrix, nodes)
                path, answer = algorithm.run()
                write_result(file_path, choice, path, answer)
            elif choice == 3 or choice == 4:
                break
            else:
                print("Please enter a valid choice\n")
                continue

        if choice == 4:
            break