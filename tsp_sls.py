import numpy as np
import math
import random
import time

class SLS:
    def __init__(self,dist_matrix,total_nodes):
        self.distance_matrix = dist_matrix
        self.total_nodes = total_nodes
        self.max_random_restarts = 500
        self.max_hill_climbing_iterations = 100
        
    def get_random_neighbor(self,neighbors):
        # Returns a random neighbor among the input list of neighbors
    
        costs_sum = 0
        for neighbor in neighbors:
            costs_sum += neighbor[1]
    
        weights = []
        for neighbor in neighbors:
            weights.append(costs_sum / neighbor[1])
    
        return random.choices(neighbors, weights, k=1)
    
    def get_cost(self,path):
        # Computes the total tour cost of input path
    
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance_matrix[int(path[i])][int(path[i + 1])]
        return cost

    def get_neighbors(self,path):
        # Takes a path(state) as input and computes all possible neighbors for it.
    
        n = len(path)
        neighbors = []
        for i in range(0, n - 2):
            for j in range(i + 2, n - 1):
                if len([path[i], path[i + 1], path[j], path[j + 1]]) <= len(set([path[i], path[i + 1], path[j], path[j + 1]])):
                    newpath = path[:i + 1] + path[i + 1:j + 1][::-1] + path[j + 1:]
                    neighbors.append([newpath,self.get_cost(newpath)])
    
        return neighbors

    def get_lesser_cost_neighbors(self,neighbors, current_state_cost):
        # Returns the neighbors with cost lesser than current state cost amongst all the neighbors
    
        output = []
        for neighbor in neighbors:
            if self.get_cost(neighbor[0]) < current_state_cost:
                    output.append(neighbor)
    
        return output if len(output) > 0 else -1
    
    def stochasticHillClimbing(self,state):
        # Function to run an iteration of stochastic hill climbing
    
        current_state_cost = self.get_cost(state)
        current_path = state
        counter = 0
        while counter < self.max_hill_climbing_iterations:
            neighbors = self.get_neighbors(path=current_path)
            less_cost_neighbors = self.get_lesser_cost_neighbors(neighbors, current_state_cost)

            # If all neighbors have tour cost > current state, we have reached a local maxima, thus exit.
            if less_cost_neighbors == -1:
                return current_path, current_state_cost

            val = self.get_random_neighbor(less_cost_neighbors)
            current_path = val[0][0]
            current_state_cost = self.get_cost(current_path)
            counter += 1

        return current_path, current_state_cost
    
    
    def getRandomStartState(self,nodes):
        # Returns a random start state for random restart wrapper
    
        path = [i for i in range(nodes)]
        random.shuffle(path)
        path.append(path[0])
        return path
    
    
    def randomRestartWrapper(self):
        # Function to call hill climbing from random start states
    
        answer = [None, math.inf]
        counter = 0
        start = time.time()
        while counter < self.max_random_restarts:
            random_path = self.getRandomStartState(self.total_nodes)
            final_path, final_cost = self.stochasticHillClimbing(random_path)

            if final_cost < answer[1]:
                answer[0] = final_path
                answer[1] = final_cost
            counter += 1
            end = time.time()

            # If runtime exceeds 600 seconds, stop
            if end-start > 600:
                break
        return answer+[end-start]

    # Run the algorithm
    def run(self):
        answer = self.randomRestartWrapper()
        path = ""
        for val in answer[0]:
            path+=str(val)+" -> "
        path = path[:-4]
        print("\nResults from Stochastic Local Search:")
        print("\nTime taken:",answer[2], "seconds")
        print("Final path:", path)
        print("Cost of final path is ",answer[1])
        return path, answer