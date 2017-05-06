
from pyspark import SparkContext
from operator import add

'''
Assignment 13 question 3
Use map and fold to calculate the average of the square root of all the numbers from 1 to 1000

'''
if __name__ == '__main__':
    sc = SparkContext("local", "squareroot_avg")
    # Create an RDD of numbers from 1 to 1,000
    nums = sc.parallelize(range(1,1001))
    # Mapping all the element to their square root
    nums_mapped = nums.map(lambda x: x**0.5)
    #Calculate the average of the square root
    sqt_sum = nums_mapped.fold(0, add) / 1000

    print("the average of the square root of all the numbers from 1 to 1000 is ", sqt_sum)
