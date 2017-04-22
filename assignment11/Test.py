
'''
Sharon Jiaqian Liu
this program is used to test parallel_sorter.py
Test to check if it is a sorted array in Process 0 and it's None otherwise.
'''
import numpy as np
import unittest
from parallel_sorter import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Test(unittest.TestCase):

    def test(self):
        res = parallel_sorter()
        if rank == 0:
            assert np.asarray(sorted(res)).all() == res.all()
        else:
            assert res == None
 

if __name__ == '__main__':
    unittest.main()


