
from itertools import *

def zbits(n, k):
    '''
    Input arguments: n is the length of all binary strings. k is the number of zero bits, one per line
    Output: a set of strings
    '''
    #First, validate the input values
    
    # if k or n is not an integer, return error message
    if int(n) != n or int(k) != k:
        raise ValueError("k and n should be integers")

    # if n <= 0, return error message
    if n <= 0:
        raise ValueError("n should not be negative or zero")
        
    # if k < 0, return error message
    if k < 0:
        raise ValueError("k should not be negative")
    

    #Then, use itertool combinations module to generate strings and add them into the result set
    result = set()
    for cp in combinations(range(n), k):
        # initialiezed a temp variable
        tmp = ["1"] * n
        # each items in cp is used as index, and set the number at the index to "0"
        for i in cp:
            tmp[i] = "0"
        result.add("".join(tmp))
    return result


# Called from terminal, all test cases are passed
'''
if __name__ == '__main__':

    assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
    assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
    assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}
'''




