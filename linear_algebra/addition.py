import numpy as np
import theano
from theano import tensor as T
import time
rng = np.random

x = T.fmatrix('x')
y = T.fmatrix('y')
z = x + y
f = theano.function([x, y], z)

print 'Matrix addition: C = A + B with matrix size (m, n)'
m = input('m = ')
n = input('n = ')

A = rng.randn(m, n).astype(np.float32) + 5 
B = rng.randn(m, n).astype(np.float32) + 10

t1 = time.time()
# theano matrix addition
C = f(A, B)
t2 = time.time()
# print C
print 'Theano matrix addition spends %f seconds.' %(t2 - t1)

t1 = time.time()
# nupmy matrix addition
D = np.add(A, B)
t2 = time.time()
# print D
print 'Numpy matrix addition spends %f seconds.' %(t2 -t1)

# Check error between numpy and theano.
print 'Check the computation result.'
print np.allclose(C, D)

