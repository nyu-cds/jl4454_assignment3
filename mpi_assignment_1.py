
# coding: utf-8

# In[5]:

from mpi4py import MPI

comm = MPI.COMM_WORLD
#get rank
rank = comm.Get_rank()

#the processes with even rank
if rank % 2 == 0:
    print("Hello from process ", rank)
#the processes with odd rank
else:
    print("Goodbye from process ", rank)


# In[ ]:



