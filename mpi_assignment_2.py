
import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#buffer initialization
temp = numpy.zeros(1)
size = comm.Get_size()

if rank == 1:
    #initialize the input variable
    inputValue=-1
    #looping until get valid input value
    while inputValue == -1 or temp[0] >= 100:
        inputValue=input("Insert an integer less than 100: ")
        try:
            temp[0]=inputValue
        except ValueError:
            print("Error, Invalid input.")
            inputValue=-1
        
        if size > 1:
            #Process 0 sends the value to process 1 which multiplies it by its rank.
            comm.Send(temp, dest=rank+1)
            #The last process sends the value back to process 0, which prints the result.
            comm.Recv(temp, source=size-1)
            
        
    print("The final result is ", temp[0])



else:
    #get value from previous rank
    comm.Recv(temp, source=rank-1)
    #multiply the value received by its rank
    print("Process ", rank, " received ", temp[0])
    temp[0] *= rank
    #if it's last process, send back to process 0
    if rank == size-1:
        comm.Send(temp, dest=0)
    else:
        comm.Send(temp, dest=rank+1)






