import numpy as np
import random as rnd


# ======================================= 1. =======================================
# LU-decomposition via partial pivot (row)
def LU_decomposition_partial_pivot(A):
    n = len(A)
    P = np.eye(n)  # permutation matrix
    L = np.eye(n)  # lower-triangular; stores multipliers
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
            U[j, i:n] -= L[j, i] * U[i, i:n]  # U[j, i] = 0 (U's j-th row and i-th column elem is eliminated/equals 0)
            U[j, i] = 0.0

    return P, L, U, permut

# Define matrices
n = rnd.randint(1, 10)
A = np.random.uniform(0, high=10.0, size=(n, n))
P, L, U, permut = LU_decomposition_partial_pivot(A)

# LU = PA
# Matrix multiplication @, np.allclose() to approximate float comparisons
print(f"1. LU = PA: {np.allclose(L@U, P@A)}")

# ========================================= a) ===========================================
# det(A) if LU = PA via partial pivot
sign = 1 if permut % 2==0 else -1  # number of swaps
detU = np.prod(np.diag(U))  # bcs U is upper-triangular
print(f"a) det(A) = {sign*detU},\n"
      f"   check via np.linalg.det(A): {np.linalg.det(A)}\n")

# ========================================= b) ==========================================
# Ax = b |*P
# PAx = Pb (PA = LU) =>
# L(Ux) = Pb
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
print(f"b) x = {x}\n\n"
      f"   Ax = b (LU = A): {np.allclose(A @ x, b)}\n")

# ======================================= c) =============================================
# Find inverse matrix for A: b = e_i => L(Ux_i) = Pe_i for each column x_i of A_inv
# Therefore, to find inverse matrix, solve one system for each basis vector
E = np.eye(n)
A_inv = np.zeros((n, n))
def find_inv(P, L, U):
    for i in range(n):
        Pe = P @ E[0:n, i]  # P * i-th column in E
        y1 = forward_substitution(L, Pe)
        x1 = backward_substitution(U, y1)
        A_inv[0:n, i] = x1

    return A_inv

print(f"c) A_inv (A^-1): \n{find_inv(P, L, U)}\n\n"
      f"   A_inv * A = A * A_inv = E: {np.allclose(A @ A_inv, E)}\n")

# ======================================= d) =========================================
# Calculate the condition number of A (for an arbitrary norm)
def matrix_norm_1(A):
    max_sum = 0  # arbitrary norm: max sum of elems in an i-th row (||A||_1 norm)
    for i in range(n):
        col_sum = sum(abs(A[:, i]))
        max_sum = col_sum if col_sum > max_sum else max_sum
    return max_sum

print(f"d) Condition number (np.linalg.norm(M, 1)): {np.linalg.norm(A, 1) * np.linalg.norm(A_inv, 1)}\n"
      f"   Calculated condition number: {matrix_norm_1(A) * matrix_norm_1(A_inv)}\n")


# ============================================= 2. ==========================================
# Find the rank of degenerate (singular) matrices
# For a singular matrix we may reach a step where there is no usable pivot left in the current column
# It can be either because: the current column is all zeros below the current row, or
#                           after elimination the remaining submatrix is all zeros
#
# For partial pivoting: move to the next column remaining in the same row
#
# The number of successful pivots is the rank
#
# Main idea: have separate indices for rows and columns
def LU_decompose_for_singular(A):
    n = len(A)
    P = np.eye(n)  # permutation matrix; fixes bad pivots (near zero/zero) by row swaps
    L = np.eye(n)
    U = A.copy()

    permut = 0

    rank = 0
    pivot_cols = []

    row_idx = 0
    col_idx = 0

    while row_idx < n and col_idx < n:
        # 1) Find pivot row index in current column, below current row
        column_i = U[row_idx:n, col_idx]

        # np.argmax() returns index of max elem in column
        pivot = np.argmax(abs(column_i)) + row_idx  # absolute row index

        # 2) If column is all zeros, skip it and move on to the next one: col_idx += 1
        if abs(U[pivot, col_idx]) < 1e-12:
            col_idx += 1
            continue  # Column is already zeroed

        # 3) If pivot row != current row (indices),
        #    swap rows in U, P and in the already computed part of L (after swap in U) (columns 0:col_idx)
        if pivot != row_idx:
            U[[row_idx, pivot], 0:n] = U[[pivot, row_idx], 0:n]
            P[[row_idx, pivot], 0:n] = P[[pivot, row_idx], 0:n]
            permut += 1
            L[[row_idx, pivot], 0:row_idx] = L[[pivot, row_idx], 0:row_idx]  # swap prev multipliers

        # After swapping the row pivot elem is U[i,i]. Columns < i are zero after swaps

        # 4) Compute multipliers and eliminate below pivot in U (elems in column i > n and rows j > i)
        for j in range(row_idx+1, n):  # for each row j > row_idx
            # How much of row i is subtracted from row j:

            if abs(U[j, col_idx]) < 1e-12:  # elem is almost zero
                continue

            L[j, row_idx] = U[j,col_idx] / U[row_idx, col_idx]  # store multiplier
            U[j, col_idx:n] -= L[j, row_idx] * U[row_idx, col_idx:n]  # U[j, col_idx] = 0 (or close to zero, so =>)
            U[j, col_idx] = 0.0

        rank += 1
        pivot_cols.append(col_idx)
        row_idx += 1
        col_idx += 1

    return P, L, U, permut, rank, pivot_cols

def backward_substitution_singular(U, y, pivot_cols):
    n = U.shape[1]  # cols
    x = np.zeros(n)

    # Consistency check for a system Ax = b
    # Provide any solution if the solution exists
    for i in range(U.shape[0]):  # rows
        # 0x1 + 0x2 + 0x3 = non zero y[i] => no solution
        if np.allclose(U[i, 0:n], 0, 1e-12) and not np.isclose(y[i], 0, 1e-12):
            return None  # no solution

    # The i-th equation row-wise: u_ii * x_i + [sum (j=i+1 in range(n)) u_ij * x_j] = y_i
    #
    # Solve only pivot variables (cols indices from pivot cols)
    #
    # Free variables stay 0
    # (ex., 0x1 + 0x2 + 0x3 = zero y[i] => third row is all zeros => x3 = 0), skip 'em

    for k in reversed(range(len(pivot_cols))):
        col_idx = pivot_cols[k]
        x[col_idx] = (y[k] - np.dot(U[k, col_idx + 1:n], x[k + 1:n])) / U[k, col_idx]

    return x

def solve_LU_singular(P, L, U, b, pivot_cols):
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = backward_substitution_singular(U, y, pivot_cols)
    return x

def generate_singular(m, k):
    # Multiply m x k matrix by k x m where m > k, the resulting matrix m x m is singular

    A = np.random.uniform(0, high=10.0, size=(n, k))
    B = np.random.uniform(0, high=10.0, size=(k, n))

    return A @ B  # singular matrix


m = rnd.randint(11, 20)
k = rnd.randint(1, 10)
S = generate_singular(m, k)

P1, L1, U1, permut1, rank, pivot_cols = LU_decompose_for_singular(S)

b1 = np.random.randn(n)
x1 = solve_LU_singular(P1, L1, U1, b1, pivot_cols)

print(f"\n2. type(x1): {type(x1)}, x1 = {x1}\n")
if type(x1) != np.ndarray:
    print("No solution: x1 does not exist (inconsistent system)\n")
else:
    print(f"Sx1 = b1: {np.allclose(S @ x1, b1)}\n")

rank_augmented = np.linalg.matrix_rank(S)
print(f"Rank: {rank}, rank of augmented S matrix: {rank_augmented}\n")

sign1 = 1 if permut % 2==0 else -1  # number of swaps
detU1 = np.prod(np.diag(U1))  # bcs U is upper-triangular
print(f"Singular matrix: det(A) = 0? {np.allclose(sign1*detU1, 0, 1e-12)}\n"
      f"check via np.linalg.det(S): {np.allclose(np.linalg.det(S), 0, 1e-12)}\n")

# ============================================= 3. ==========================================

# Householder reflections to make QR-decomposition
#
# Zero out everything below the main diag in a column
# v defines a reflection
#
# The reflector is H = I (identity matrix) - 2 * v * v^T (with a normalized v)
# HR -> R, QH -> Q because QR by Householder repeatedly applies reflectors
# The H matrix has the property hat it reflects x onto the first coordinate axis:
#       Hx = alpha * e1 = [alpha, 0, ..., 0]^T (it is a column)
# (that's why all entries below the diagonal become zero)
def QR_Householder(A):
    n = len(A)
    Q = np.eye(n)  # orthogonal matrix we accumulate, so that A = QR
                   # (remember transformations that made A into R)
    R = A.copy()  # upper-triangular: simplified A

    for i in range(n):  # for each column
        # 1) Extract the column vector from i to n row, that we need to clean
        #    Take the part below main diag, including it
        x = R[i:n, i]

        # 2) Build the Householder reflector H
        #    v = +- (x - alpha*e1) / ||(x - alpha*e1)||
        #    e1 = [1, 0, ..., 0]^T (it is a column), alpha = -sgn(x) * ||x||
        e1 = np.zeros_like(x)
        e1[0] = 1.0

        alpha = np.linalg.norm(x)
        if x[0] >= 0:
            alpha = -alpha  # to avoid catastrophic cancellation in calculation of v

        v = x - alpha * e1
        v = v / np.linalg.norm(v)  # normalize v (it's a unit vector)

        # 3) Apply reflector H to R to zero out entries below the diagonal in column i
        #    Only apply to submatrix to avoid touching previous columns (i:n)
        #    Matrix R is being triangularized (apply the reflector) from the left HR -> R
        #    HR -> R changes rows of the active block
        #    R = HR = (I - 2 * v * v^T)R = R - 2 * v * (R * v^T)
        R[i:n, i:n] -= 2 * np.outer(v, v @ R[i:n, i:n])

        # 4) Accumulate Q: multiply all reflectors together to form Q
        #
        #    Apply only to relevant columns (i:n)
        #    (but all rows bcs multiplying on the right changes cols
        #    and every row participates in those cols)
        #
        #    QH -> Q changes columns of the active block
        #
        #    Apply the reflector from the right QH -> Q
        #    Q = QH = Q(I - 2 * v * v^T) = Q - 2 * (Q * v) * v^T
        #
        #    Q[:, i:] is a matrix [n x n-i], v is a vector [n-i] =>
        #    Q @ v's result is a column vector [n] w
        #    outer product of two vectors w and v is a matrix [n x n-i]
        #
        #    Q[:, i:] @ v projects columns of Q onto direction v (how much each row aligns with that direction)
        #    np.outer() spreads that projection back across columns (rebuild matrix from that alignment)
        #    Subtraction reflects those columns across hyperplane
        Q[0:n, i:n] -= 2 * np.outer(Q[0:n, i:n] @ v, v)

    return Q, R

Q, R = QR_Householder(A)
print(f"3. LU = PA: {np.allclose(L@U, P@A)}")
print(f"QR = A: {np.allclose(Q@R, A)}\n")

# Bcs Q is orthogonal and R is upper-triangular:
# Ax = b => Q(Rx) = b => Rx = Q^T * b
def solve_QR(Q, R, b):
    y = Q.T @ b
    x = backward_substitution(R, y)
    return x

print(f"x = {x}\n\n"
      f"Ax = b (QR = A): {np.allclose(A @ x, b)}\n")
