from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["N", "H_Lower_Bound" ,"Min_Path_Cost","RunTime"]
df = pd.read_csv("BNBvSLS_bnb.csv", names=columns)
plt.plot(df.N,df.H_Lower_Bound,label = 'Heuristic Lower Bound')
plt.plot(df.N,df.Min_Path_Cost, label = 'Minimum Cost Path')
plt.xlabel('Number of Nodes')
plt.ylabel('Runtime in Seconds')
plt.title('Heuristic Analysis')
plt.legend()
plt.show()