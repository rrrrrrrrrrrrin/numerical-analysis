import numpy as np
import random as rnd
import math


# LU-decomposition via partial pivot (row)
def LU_decomposition_partial_pivot(A):
    n = len(A)
    P = np.eye(n)  # permutation matrix; fixes bad pivots (near zero/zero) by row swaps
    L = np.eye(n)  # lower triangular; stores multipliers
    U = A.copy()  # upper-triangular; the result of Gaussian elimination (simplified A)

    permut = 0

    # Algorithm: Gaussian elimination with row pivoting
    for i in range(n):  # for each column
        # 1) Find pivot row index
        column_i = U[i:n, i]  # U's i-th column from row i to n

        # np.argmax() returns index of max elem in column
        pivot = np.argmax(abs(column_i)) + i  # absolute row index

        # 2) Check pivot's value
        if abs(U[pivot, i]) < 1e-12:
            continue  # Column is already zeroed

        # 3) If pivot row != current row (indices),
        #    swap rows in U, P and in the already computed part of L (after swap in U) (columns 0:i)
        if pivot != i:
            U[[i, pivot], 0:n] = U[[pivot, i], 0:n]
            P[[i, pivot], 0:n] = P[[pivot, i], 0:n]
            permut += 1
            L[[i, pivot], 0:i] = L[[pivot, i], 0:i]  # swap prev multipliers

        # After swapping the row pivot elem is U[i,i]. Columns < i are zero after swaps

        # 4) Compute multipliers and eliminate below pivot in U (elems in column i > n and rows j > i)
        for j in range(i+1, n):  # for each row j > i
            # How much of row i is subtracted from row j:
            L[j,i] = U[j,i] / U[i,i]  # store multiplier
            U[j, i:n] -= L[j, i] * U[i, i:n]  # U[j, 0] = 0 (U's j row and i column elem is eliminated/equals 0)

    return P, L, U, permut

# Define matrices
n = rnd.randint(0, 100)
A = np.random.uniform(0, high=10.0, size=(n, n))
P, L, U, permut = LU_decomposition_partial_pivot(A)

# 1. LU = PA
# Matrix multiplication @, np.allclose() to approximate float comparisons
print(f"1. LU = PA: {np.allclose(L@U, P@A)}")

# a) det(A) if LU = PA via partial pivot
sign = 1 if permut % 2==0 else -1  # number of swaps
detU = np.prod(np.diag(U))  # bcs U is upper-triangular
print(f"  a) det(A) = {sign*detU},\n     "
      f"check via np.linalg.det(A): {np.linalg.det(A)}\n")

# b) Ax = b |*P
#    PAx = Pb (PA = LU) =>
#    L(Ux) = Pb
# Let: Ux = y, Ly = Pb

def forward_substitution(L, Pb):
    n = len(L)  # L[i,i] = 1
    y = np.zeros(n)

    # The i-th equation row-wise: l_ii * y_i + [sum (j=0 in range(i)) l_ij * y_j] = (Pb)_i

    for i in range(n):
        # np.dot() computes the sum (j=0 in range(i)) l_ij * y_j
        y[i] = Pb[i] - np.dot(L[i, 0:i], y[0:i])  # L[i,i] = 1 => no division

    return y

def backward_substitution(U, y):
    n = len(U)
    x = np.zeros(n)

    # The i-th equation row-wise: u_ii * x_i + [sum (j=i+1 in range(n)) u_ij * x_j] = y_i

    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i + 1:n], x[i + 1:n])) / U[i, i]

    return x

def solve_LU(P, L, U, b):
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = backward_substitution(U, y)
    return x

b = np.random.randn(n)
x = solve_LU(P, L, U, b)
print(f"  b) x = {x}\n"
      f"  Ax = b: {np.allclose(A @ x, b)}")
