
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

