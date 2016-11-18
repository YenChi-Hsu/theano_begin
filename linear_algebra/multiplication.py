import numpy as np
import theano
from theano import tensor as T
import time
rng = np.random

x = T.fmatrix('x')
y = T.fmatrix('y')
z = T.dot(x, y)
f = theano.function([x, y], z)

print 'Matrix multiplication: C = A + B with matrix size (m, n)'
m = input('m = ')
n = input('n = ')

A = rng.randn(m, n).astype(np.float32) + 5 
B = rng.randn(n, m).astype(np.float32) + 10

t1 = time.time()
# theano matrix addition
C = f(A, B)
t2 = time.time()
# print C
print 'Theano matrix multiplication spends %f seconds.' %(t2 - t1)

t1 = time.time()
# nupmy matrix addition
D = np.dot(A, B)
t2 = time.time()
# print D
print 'Numpy matrix multiplication spends %f seconds.' %(t2 -t1)

# Check error between numpy and theano.
print 'Check the computation result.'
print np.allclose(C, D)

