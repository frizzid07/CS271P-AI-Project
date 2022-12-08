import numpy as np
from collections import deque
import time

# This class defines the attributes of every node and the heuristics to calculate
class Node:
    def __init__(self, label, matrix=np.array([]), cost=0):
        self.label = label
        self.matrix = matrix
        self.cost = cost

    def __iter__(self):
        yield self.label

    def __repr__(self):
        return str(self.label)

    def update_matrix(self, matrix, current, parent):
        matrix[parent] = np.inf
        matrix[:, current] = np.inf
        matrix[current, parent] = np.inf
        return matrix
    
    def reduce_matrix(self, matrix, cost, matrix_size):
        for row in range(matrix_size):
            if not np.array_equal(matrix[row], np.array([np.inf]*matrix_size)):
                min_cost = np.inf
                for col in range(matrix_size):
                    if matrix[row, col] == 0:
                        min_cost = 0
                        break
                    if min_cost > matrix[row, col]:
                        min_cost = matrix[row, col]
                if min_cost != 0 or min_cost != np.inf:
                    matrix[row] -= min_cost
                    cost += min_cost
        
        for col in range(matrix_size):
            if not np.array_equal(matrix[:, col], np.array([np.inf]*matrix_size)):
                min_cost = np.inf
                for row in range(matrix_size):
                    if matrix[row, col] == 0:
                        min_cost = 0
                        break
                    if min_cost > matrix[row, col]:
                        min_cost = matrix[row, col]
                if min_cost != 0 or min_cost != np.inf:
                    matrix[:, col] -= min_cost
                    cost += min_cost
        
        return matrix, cost

class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.start = Node(0)
        self.nodes = [Node(x) for x in range(1, self.num_nodes)]
        self.lower_bound, self.upper_bound = -np.inf, np.inf
        self.path = deque([self.start], maxlen=self.num_nodes)
        self.nodes_explored = 0

    def node_dfs(self, node, start_time, parent=None, path=deque([]), visited=set()):
        if time.time()-start_time >= 600:
            raise TimeoutError

        if node.cost >= self.upper_bound:
            return
        
        self.nodes_explored += 1
        if len(visited) == self.num_nodes-1:
            self.upper_bound = node.cost
            self.path.clear()
            self.path.append(self.start)
            temp = path.copy()
            while temp:
                self.path.append(temp.popleft())
            temp.clear()
            return

        for neighbor in self.nodes:
            if neighbor not in visited:
                neighbor.cost = node.cost + node.matrix[node.label, neighbor.label]
                neighbor.matrix = neighbor.update_matrix(node.matrix.copy(), neighbor.label, node.label)
                neighbor.matrix, neighbor.cost = neighbor.reduce_matrix(neighbor.matrix, neighbor.cost, self.num_nodes)
                visited.add(neighbor)
                path.append(neighbor)
                try:
                    self.node_dfs(neighbor, start_time, node, path, visited)
                except:
                    break
                visited.remove(neighbor)
                path.pop()

    def init_dfs(self, start_time):
        np.fill_diagonal(self.start.matrix, np.inf)
        self.start.matrix, self.start.cost = self.start.reduce_matrix(self.start.matrix, 0, self.num_nodes)
        self.lower_bound = self.start.cost
        self.node_dfs(self.start, start_time)

    # Run the algorithm
    def run(self):
        start_time = time.time()
        self.init_dfs(start_time)
        end_time = time.time()
        print("\nResults for Branch and Bound:")
        print("\nTime taken:", end_time-start_time, "seconds")
        print("Lower Bound from Heuristic function:", self.lower_bound)
        print("Total number of nodes explored:", self.nodes_explored)
        print("\nFinal path:", end=" ")
        output = self.path.copy()
        while output:
            print(output.popleft(), end=" -> ")
        print(self.start.label)
        output.clear()
        print("Cost of final path: ", self.upper_bound)
        return end_time-start_time