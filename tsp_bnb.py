import numpy as np
from collections import deque
import time

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
        self.nodes = [Node(x) for x in range(1, num_nodes)]
        self.lower_bound, self.upper_bound = -np.inf, np.inf
        self.path = deque([self.start], maxlen=self.num_nodes)

    def node_dfs(self, node, start_time, parent=None, path=deque([]), visited=set()):
        if time.time()-start_time >= 600:
            raise TimeoutError

        if node.cost >= self.upper_bound:
            return
        
        if len(visited) == self.num_nodes-1:
            self.upper_bound = node.cost
            # print("New Upper Bound:", self.upper_bound)
            self.path.clear()
            self.path.append(self.start)
            temp = path.copy()
            while temp:
                self.path.append(temp.popleft())
            # print("Updated Path:", self.path)
            temp.clear()
            return

        for neighbor in self.nodes:
            if neighbor not in visited:
                neighbor.cost = node.cost + node.matrix[node.label, neighbor.label]
                neighbor.matrix = neighbor.update_matrix(node.matrix.copy(), neighbor.label, node.label)
                neighbor.matrix, neighbor.cost = neighbor.reduce_matrix(neighbor.matrix, neighbor.cost, self.num_nodes)
                visited.add(neighbor)
                # print("Visited:", visited)
                path.append(neighbor)
                # print("Path so far:", path)
                try:
                    self.node_dfs(neighbor, start_time, node, path, visited)
                except:
                    break
                visited.remove(neighbor)
                path.pop()

    def init_dfs(self, start_time):
        self.start.matrix, self.start.cost = self.start.reduce_matrix(self.start.matrix, 0, self.num_nodes)
        self.lower_bound = self.start.cost
        # print("Lower Bound:", self.lower_bound)
        # print("Upper Bound:", self.upper_bound)
        # print(self.start.label, self.start.matrix, self.start.cost)
        self.node_dfs(self.start, start_time)

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
        f"{'BNBvsSLS'}_{n}.out",
        distance_matrix,
        delimiter=" ",
        fmt="%1.4f",
        header=str(n),
        comments="",
    )

    return distance_matrix, f"{n}_{mean}_{sigma}.out"

def write_result(file_path, graph, time_taken):
    with open(file_path, "a") as f:
        f.write("\nHeuristic Lower Bound: " +str(graph.lower_bound))
        f.write("\nMinimum Path Cost: " +str(graph.upper_bound))
        f.write("\nPath: ")
        while graph.path:
            f.write(str(graph.path.popleft()) + " -> ")
        f.write(str(graph.start.label))
        f.write("\nTime taken: " +str(time_taken)+ " seconds")

if __name__ == "__main__":
    
    import csv
    headers = ['N','Heuristic Lower Bound','Min_Path_Cost','Runtime']
    file = open('BNBvSLS_bnb.csv','w')
    writer = csv.writer(file)

    writer.writerow(headers)

    for i in range(4,20):

        nodes = i
        mean = 100
        sigma = 30
        dist_matrix, file_path = write_distance_matrix(nodes, mean, sigma)
        np.fill_diagonal(dist_matrix, np.inf)

        graph = Graph(nodes)
        graph.start.matrix = dist_matrix

        start_time = time.time()
        graph.init_dfs(start_time)

        print("\nHeuristic Lower Bound: ", graph.lower_bound)
        print("Minimum Path Cost: ", graph.upper_bound)
        print("Path: ", end="")
        while graph.path:
            print(graph.path.popleft(), end=" -> ")
        print(graph.start.label)
        print("Time taken: ", time.time()-start_time, " seconds")

        data = [i,graph.lower_bound,graph.upper_bound,time.time()-start_time]
        writer.writerow(data)
        file.flush()
    
    file.close()