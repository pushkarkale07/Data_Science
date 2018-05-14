#-------------------------------------------------------------------------
'''
    Problem 1: getting familiar with python and unit tests.
    In this problem, please install python verion 2.7 and the following package:
        * nose   (for unit tests)

    To install python packages, you can use any python package managment software, such as pip, conda. For example, in pip, you could type `pip install nose` in the terminal to install the package.

    Then start implementing function swap().
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def bubblesort( A ):
    for i in range(len(A)):
        for k in range( len(A) - 1, i, -1):
            if ( A[k] < A[k - 1] ):
                swap(A, k, k - 1)

def swap( A, i, j ):
	temp = A[j]
	A[j] = A[i]
	A[i] = temp

