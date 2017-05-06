from pyspark import SparkContext

import re
'''
Assignment 13 question 1
    
First, remove any non-words and split lines into separate words. Follow mapping and reduction with my
own function. Finally, count the distinct words
'''
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

def distfunction(x, y):
    '''reduction function: simplely aggregate keys to finds the distinct words'''
    return 1

if __name__ == '__main__':
    sc = SparkContext("local", "distinct_word_count")
        
    text = sc.textFile('pg2701.txt')
    words = text.flatMap(splitter)
    words_mapped = words.map(lambda x: (x,1))
    sorted_map = words_mapped.sortByKey()
    #reduction by distfunction
    counts = sorted_map.reduceByKey(distfunction)
    
    print("the number of distinct words is ",counts.count())
