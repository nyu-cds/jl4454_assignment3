from pyspark import SparkContext
from operator import mul


'''
Assignment 13 question 2
Use 'fold' method to creat a program that calculates the product of all the
numbers from 1 to 1000 and prints the result.

'''
if __name__ == '__main__':
    sc = SparkContext("local", "product_spark")
    # Create an RDD of numbers from 1 to 1,000
    nums = sc.parallelize(range(1,1001))
    # Use fold method to calculate. Because we use multiplication, so the zeroValue is 1.
    print("the product of all the numbers is ", nums.fold(1, mul))
