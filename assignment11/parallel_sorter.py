
#Sharon Jiaqian Liu
#assignment 11
import numpy as np
import random
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#the length of the random array
N = 10000

def parallel_sorter():
    data_final=[]
    #the process 0 will generate the unsorted array with N elements and split between processes
    if rank == 0:
        data = np.random.randint(0, N, N)
        #Find the max and min number of the array and the split points
        max_n = max(data)
        min_n = min(data)
        split_pt = (max_n-min_n)/(size-1)
        
        #slice the array
        
        for i in range(size):
            #add the element in each range
            data_final.append(data[(data>min_n+split_pt*(i-1)) & (data<=min_n+split_pt*(i))])
    else:
        #if this's not process 0, the data_final shouldn't make sense
        data_final = None
    
    # Send to the processes
    data_sliced = comm.scatter(data_final, root=0)

    # Sort data in each process except the process 0
    if rank != 0:
        data_sliced.sort()

    # Catch sorted data
    data_sorted = comm.gather(data_sliced, root=0)

    # Process 0 produces a completely sorted data set
    if rank == 0:
        data_sorted = np.concatenate(data_sorted)

    return data_sorted

if __name__ == '__main__':
    res = parallel_sorter()
    print("the result is ", res)



