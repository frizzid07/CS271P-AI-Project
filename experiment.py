from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["N", "H_Lower_Bound" ,"Min_Path_Cost","RunTime"]
df = pd.read_csv("BNBvSLS_bnb.csv", names=columns)
plt.plot(df.N,df.)
plt.xlabel('Number of Nodes')
plt.ylabel('Runtime in Seconds')
plt.title('RuntTime analysis of SLS algorithm')
plt.show()