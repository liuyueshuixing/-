import torch
import numpy as np
index = np.array([[0,0],[1,1],[1,2]])
value = np.array([0.5,0.7,0.3])
shape = (3,3)
def sparse_to_dense(sparse):
    count = 0
    metrics = np.zeros(sparse[2])
    for index in sparse[0]:
        row = int(index[0])
        col = int(index[1])
        metrics[row][col] = sparse[1][count]
        count = count + 1
    return metrics
sparse = (index,value,shape)
sparse_to_dense(sparse)
print()