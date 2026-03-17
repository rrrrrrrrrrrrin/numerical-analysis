import numpy as np
import random as rnd
import math

# LU-decomposition via partial pivot (row)
def LU_decomposition_partial_pivot(A):
    n, m = A.shape
    P = np.eye(n)  # row permutation matrix
    L = np.eye(n)
    U = A.copy()

    for i in range(min(n, m)):
        # Find pivot row index
        column_i = U[i:n, i]  # U's i-th column from row i to n
        pivot = np.argmax(abs(column_i)) + i  # absolute row index

        # Check pivots's value
        if abs(U[pivot, i]) < 1e-12:
            continue  # Column is already zeroed

        # If pivot row != current row (indices),
        # swap rows in U, P and the multipliers already in L
        if pivot != i:
            U[[i, pivot], :] = U[[pivot, i], :]
            P[[i, pivot], :] = P[[pivot, i], :]
            L[[i, pivot], 0:i] = L[[pivot, i], 0:i]  # swap prev multipliers

        # Compute multipliers and eliminate below pivot in U
        for j in range(i+1, n):
            L[j,i] = U[j,i] / U[i,i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U

# Define matrices
n = rnd.randint(0, 100); m = rnd.randint(0, 100)
A = np.random.uniform(0, high=10.0, size=(n, m))
P, L, U = LU_decomposition_partial_pivot(A)

# Matrix multiplication @, np.allclose() to approximate float comparisons
# LU = PA
print(np.allclose(L@U, P@A))