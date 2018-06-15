
# -*- coding: utf-8 -*-
"""
Python Comment Block

"""

# Python Comment
#import openpyxl

#Basic linear algebra library
# Already included in NumPy

a=[1,2,3]
b=[-1,-2,-3]
c=[3,4,5]
A=[a,b,c]

def vector_add(v, w):
    # adds corresponding elements
    return [v_i + w_i
            for v_i, w_i in zip(v,w)]
    
def vector_subtract(v, w):
    # subtracts corresponding elements
    return [v_i - w_i
            for v_i, w_i in zip(v,w)]
    
def vector_multiply(v, w):
    # subtracts corresponding elements
    return [v_i * w_i
            for v_i, w_i in zip(v,w)]
    
vector_multiply([1,2,"a"],[4,5,6])

def scalar_multiply(c, v):
    # multiplies a scalar with a vector
    return [c * v_i
            for v_i in v]
    
def vector_sum(vectors):
    #sums the individual elements across vectors, returning a vector
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

import functools #reduce requires functools in Python 3
def vector_sum2(vectors):
    return functools.reduce(vector_add, functools.reduce)    
    
def vector_mean(vectors):
    #compute the vector whos ith element is the mean of the ith elments of the
    # input vectors
    n = len(vectors)
    result = vector_sum(vectors)
    result = scalar_multiply(1/n, result)
    return result
    
def dot(v, w):
    #returns the dot product of two vectors
    return sum(v_i * w_i
            for v_i, w_i in zip(v,w))

def sum_of_squares(v):
    #returns the component wise square of v
    return dot(v,v)

import math
def magnitude(v):
    #returns the scalar magnitude of a vector v
    return math.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    #returns the squared L2 distance between two points
    return sum((v_i - w_i) ** 2  for v_i, w_i in zip(v,w))

def distance(v, w):
    #returns the L2 distance between two points
    return math.sqrt(squared_distance(v,w))

def shape(A):
    #returns the number of rows and columns in matrix A
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 #this is single line syntax for ifelse
    return num_rows, num_cols

def get_row(A, i):
    #returns the ith row of Matrix A
    return A[i]

def get_column(A, j):
    #returns the jth column of matrix A
    return [A_i[j]
            for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    #returns a num_rows x num_cols matrix whose (i,j)th element = f(i,j)                
    return [[entry_fn(i,j) 
                for j in range(num_cols)]
                for i in range(num_rows)]

def test_matrix(i,j):
    return (i,j)

make_matrix(3,5,test_matrix)

def is_diagonal(i,j):
    return 1 if i == j else 0

def identity(n):
    #returns an identity matrix of size nxn
    return make_matrix(n,n,is_diagonal)

##Summary statistics functions
def mean(x):
    return sum(x)/len(x)

def median(x):
    n = len(x)
    sorted_x = sorted(x)
    midpoint = n // 2
    
    if n % 2 == 1:
        #if the length is odd, just return the middle value
        return sorted_x[midpoint]
    else:
        #if the length is even, return the average of the two middle values
        lo = midpoint -1
        hi = midpoint
        return ((sorted_x[lo]+sorted_x[hi]) / 2)
    
def quantile(x,p):
    #returns the p-th percentile value in x
    p_index = int(p*len(x))
    return sorted(x)[p_index]

import collections
def mode(x):
    #returns a list, might be more than one mode
    counts = collections.Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

def data_range(x):
    return max(x) - min(x)


def de_mean(x):
    #returns the deviations from the mean (x-E(x))
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    #this is sample variance
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations)/(n-1)

def standard_deviation(x):
    return math.sqrt(variance(x))    
    
def interquartile_range(x):
    return quantile(x,0.75) - quantile(x, 0.25)

def covariance(x,y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)


def correlation(x,y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x,y)/(stdev_x*stdev_y)
    else:
        return 0 #technically it's undefined, but we return 0
    
def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    if x < 0 : return 0
    elif x < 1 : return x
    else: return 1
    
def normal_pdf(x, mu = 0, sigma = 1):
    sqrt_two_pi = math.sqrt(2*math.pi)
    return (math.exp(-(x-mu) ** 2 / 2  / sigma ** 2)/(sqrt_two_pi*sigma))

def normal_cdf(x, mu = 0 , sigma = 1):
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma)) /2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    #find approximate inverse using binary search
    if mu != 0 or sigma !=1:
        return mu + sigma * inverse_normal_cdf(p, tolerance = tolerance)
    low_z, low_p = -10.0,0
    hi_z, hi_p = 10.0,1
    while hi_z - low_z > tolerance:
        mid_z = (low_z+hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p
        else: 
            break
    return mid_z

import random
def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist(p, n, num_points):
    #use a bar chart to show actual binomial samples
    return None
    

        
