import numpy as np
import time

# a) Solve system of nonlinear algebraic equations via Newton's method
#    Count amount of iterations and arithmetic operations required for calculation
#    Count execution time

# System F(x) = 0
# F(x) = [f_1(x_1, ..., x_10), ..., f_10(x_1, ..., x_10)],
# where f_i is nonlinear equation
def F(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x
    return np.array([
        np.cos(x2 * x1) - np.exp(-3 * x3) + x4 * x5**2 - x6 - np.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
        np.sin(x2 * x1) + x3 * x9 * x7 - np.exp(-x10 + x6) + 3 * x5**2 - x6 * (x8 + 1) + 10.886272036407019994,
        x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
        2 * np.cos(-x9 + x4) + x5 / (x3 + x1) - np.sin(x2**2) + np.cos(x7 * x10)**2 - x8 - 0.1707472705022304757,
        np.sin(x5) + 2 * x8 * (x3 + x1) - np.exp(-x7 * (-x10 + x6)) + 2 * np.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
        np.exp(x1 - x4 - x9) + x5**2 / x8 + np.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
        x2**3 * x7 - np.sin(x10 / x5 + x8) + (x1 - x6) * np.cos(x4) + x3 - 0.7380430076202798014,
        x5 * (x1 - 2 * x6)**2 - 2 * np.sin(-x9 + x3) + 1.5 * x4 - np.exp(x2 * x7 + x10) + 3.5668321989693809040,
        7 / x6 + np.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
        x10 * x1 + x9 * x2 - x8 * x3 + np.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096
    ], dtype=float)

# Jacobian matrix J(x)[10 x 10] = [df_i/dx_j] (partial derivatives)
def J(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x
    A = np.zeros((10, 10), dtype=float)

    # df_1/dx_j, j=[1, 10]
    A[0, 0] = -np.sin(x2 * x1) * x2  # df_1/dx_1
    A[0, 1] = -np.sin(x2 * x1) * x1
    A[0, 2] = 3 * np.exp(-3 * x3)
    A[0, 3] = x5**2
    A[0, 4] = 2 * x4 * x5
    A[0, 5] = -1
    A[0, 7] = -2 * np.cosh(2 * x8) * x9
    A[0, 8] = -np.sinh(2 * x8)
    A[0, 9] = 2

    A[1, 0] = np.cos(x2 * x1) * x2
    A[1, 1] = np.cos(x2 * x1) * x1
    A[1, 2] = x9 * x7
    A[1, 4] = 6 * x5
    A[1, 5] = -np.exp(-x10 + x6) - (x8 + 1)
    A[1, 6] = x3 * x9
    A[1, 7] = -x6
    A[1, 8] = x3 * x7
    A[1, 9] = np.exp(-x10 + x6)

    A[2, :] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    A[3, 0] = -x5 / (x3 + x1)**2
    A[3, 1] = -2 * x2 * np.cos(x2**2)
    A[3, 2] = -x5 / (x3 + x1)**2
    A[3, 3] = -2 * np.sin(-x9 + x4)
    A[3, 4] = 1 / (x3 + x1)
    A[3, 6] = -2 * x10 * np.sin(x7 * x10) * np.cos(x7 * x10)
    A[3, 7] = -1
    A[3, 8] = 2 * np.sin(-x9 + x4)
    A[3, 9] = -2 * x7 * np.sin(x7 * x10) * np.cos(x7 * x10)

    A[4, 0] = 2 * x8
    A[4, 1] = -2 * np.sin(x2)
    A[4, 2] = 2 * x8
    A[4, 3] = 1 / (-x9 + x4)**2
    A[4, 4] = np.cos(x5)
    A[4, 5] = x7 * np.exp(-x7 * (-x10 + x6))
    A[4, 6] = -(x10 - x6) * np.exp(-x7 * (-x10 + x6))
    A[4, 7] = 2 * (x3 + x1)
    A[4, 8] = -1 / (-x9 + x4)**2
    A[4, 9] = -x7 * np.exp(-x7 * (-x10 + x6))

    A[5, 0] = np.exp(x1 - x4 - x9)
    A[5, 1] = -1.5 * x10 * np.sin(3 * x10 * x2)
    A[5, 2] = -x6
    A[5, 3] = -np.exp(x1 - x4 - x9)
    A[5, 4] = 2 * x5 / x8
    A[5, 5] = -x3
    A[5, 7] = -x5**2 / x8**2
    A[5, 8] = -np.exp(x1 - x4 - x9)
    A[5, 9] = -1.5 * x2 * np.sin(3 * x10 * x2)

    A[6, 0] = np.cos(x4)
    A[6, 1] = 3 * x2**2 * x7
    A[6, 2] = 1
    A[6, 3] = -(x1 - x6) * np.sin(x4)
    u = x10 / x5 + x8
    A[6, 4] = x10 / x5**2 * np.cos(u)
    A[6, 5] = -np.cos(x4)
    A[6, 6] = x2**3
    A[6, 7] = -np.cos(u)
    A[6, 9] = -np.cos(u) / x5

    A[7, 0] = 2 * x5 * (x1 - 2 * x6)
    A[7, 1] = -x7 * np.exp(x2 * x7 + x10)
    A[7, 2] = -2 * np.cos(-x9 + x3)
    A[7, 3] = 1.5
    A[7, 4] = (x1 - 2 * x6)**2
    A[7, 5] = -4 * x5 * (x1 - 2 * x6)
    A[7, 6] = -x2 * np.exp(x2 * x7 + x10)
    A[7, 8] = 2 * np.cos(-x9 + x3)
    A[7, 9] = -np.exp(x2 * x7 + x10)

    A[8, 0] = -3
    A[8, 1] = -2 * x8 * x10 * x7
    A[8, 3] = np.exp(x5 + x4)
    A[8, 4] = np.exp(x5 + x4)
    A[8, 5] = -7 / x6**2
    A[8, 6] = -2 * x2 * x8 * x10
    A[8, 7] = -2 * x2 * x10 * x7
    A[8, 8] = 3
    A[8, 9] = -2 * x2 * x8 * x7

    A[9, 0] = x10
    A[9, 1] = x9
    A[9, 2] = -x8
    A[9, 3] = np.cos(x4 + x5 + x6) * x7
    A[9, 4] = np.cos(x4 + x5 + x6) * x7
    A[9, 5] = np.cos(x4 + x5 + x6) * x7
    A[9, 6] = np.sin(x4 + x5 + x6)
    A[9, 7] = -x3
    A[9, 8] = x2
    A[9, 9] = x1

    return A

# LU-decomposition via partial pivot (row)
def LU_decomposition_partial_pivot(A):
    n = len(A)
    P = np.eye(n)  # permutation matrix
    L = np.eye(n)  # lower-triangular; stores multipliers
    U = A.copy().astype(float)  # upper-triangular; the result of Gaussian elimination (simplified A)

    permut = 0
    operations = 0

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
            operations += 1

            U[j, i:n] -= L[j, i] * U[i, i:n]  # U[j, i] = 0 (U's j-th row and i-th column elem is eliminated/equals 0)
            operations += 2 * (n - i)  # one mult, one subtraction for n - i cols

            U[j, i] = 0.0

    return P, L, U, permut, operations


# Ax = b |*P
# PAx = Pb (PA = LU) =>
# L(Ux) = Pb
# Let: Ux = y, Ly = Pb

def forward_substitution(L, Pb):
    n = len(L)  # L[i,i] = 1
    y = np.zeros(n)
    operations = 0

    # The i-th equation row-wise: l_ii * y_i + [sum (j=0 in range(i)) l_ij * y_j] = (Pb)_i

    for i in range(n):
        # np.dot() computes the sum (j=0 in range(i)) l_ij * y_j
        y[i] = Pb[i] - np.dot(L[i, 0:i], y[0:i])  # L[i,i] = 1 => no division
        operations += 2 * i  # row i contains i elems below diag (mult and sub)

    return y, operations

def backward_substitution(U, y):
    n = len(U)
    x = np.zeros(n)
    ops = 0

    # The i-th equation row-wise: u_ii * x_i + [sum (j=i+1 in range(n)) u_ij * x_j] = y_i

    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i + 1:n], x[i + 1:n])) / U[i, i]
        ops += 2 * (n - i - 1) + 1  # row i has n - i - 1 elems above diag (mult and sub) + div

    return x, ops

def solve_LU(P, L, U, b):
    Pb = P @ b
    y, ops1 = forward_substitution(L, Pb)
    x , ops2 = backward_substitution(U, y)
    return x, ops1 + ops2

# Suppose we already have an approximation x_k
#
# Near this point, approximate F(x) by its 1st-order Taylor expansion:
# F(x) = F(x_k) + F'(x_k)(x-x_k) + 1/2 * (F''(x_k)(x-x_k)^2) + ...
#
# Replace: x = x_k + dx. Therefore (x - x_k) -> (x_k + dx - x_k) -> dx
# => F(x_k + dx) = F(x_k) + F'(x_k)dx + 1/2 * (F''(x_k)dx^2) + ...
#
# Assume dx is very small; dx^2, dx^3, ... become negligible =>
# 1st-order: F(x_k + dx) =~ F(x_k) + F'(x_k)dx
#
# Suppose x is a vector. Then F'(x_k) becomes a Jacobian matrix J(x_k)
# Final form: F(x_k + dx) =~ F(x_k) + J(x_k) * dx  (IT'S Newton's method)
#
# We want next approximation x_(k+1) to satisfy F(x_(x+1)) =~ 0
# Substitute x_(k+1) = x_k + dx_k
# Therefore F(x_k) + J(x_k) * dx_k = 0
# So at each iteration we solve the linear system J(x_k) * dx_k = -F(x_k)

# Newton's method
def newton_method(x0, eps=1e-8, max_iter=50):
    x = x0.copy().astype(float)
    ops_total = 0
    iters = 0

    t0 = time.perf_counter()

    for i in range(max_iter):
        Fx = F(x)

        if not np.isfinite(Fx).all():
            print("Non-finite values in F(x)")
            break

        # The goal is to make all components of Fx as close to 0 as possible (F(x) = 0)
        # Compute infinity form for Fx (largest absolute value of vector elems)
        # Stop bcs the system is solved with required accuracy (at least eps)
        if np.linalg.norm(Fx, ord=np.inf) < eps:
            break

        # J(x_k) * dx_k = -F(x_k) is solved using LU-decomposition: PA = LU
        # Ax = b; A = J(x_k), x = dx_k b = -Fx_k
        P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
        dx, ops_solve = solve_LU(P, L, U, -Fx)

        x += dx  # x_(k+1) = x_k + dx_k
        ops_total += ops_LU + ops_solve
        iters += 1

    elapsed = time.perf_counter() - t0
    return x, iters, ops_total, elapsed

x0 = np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5], dtype=float)

x, iters, ops, elapsed = newton_method(x0)

print("a) Newton's method")
print(f'   x = {x}')
print(f'   iterations = {iters}')
print(f'   amount of arithmetic operations = {ops}')
print(f'   elapsed time = {elapsed}')
print(f'    residual norm =, {np.linalg.norm(F(x), ord=np.inf)}')

# b) Modified Newton's method via Jacobian matrix
def modified_newton_method(x0, eps=1e-8, max_iter=500):
    x = x0.copy().astype(float)
    ops_total = 0
    iters = 0

    t0 = time.perf_counter()

    # Jacobian matrix is computed only once at initial point x = x0
    P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
    ops_total += ops_LU

    for i in range(max_iter):
        Fx = F(x)

        if not np.isfinite(Fx).all():
            print("Non-finite values in F(x)")
            break

        if np.linalg.norm(Fx, ord=np.inf) < eps:  # too small
            break

        dx, ops_solve = solve_LU(P, L, U, -Fx)

        x += dx
        ops_total += ops_solve
        iters += 1

    elapsed = time.perf_counter() - t0
    return x, iters, ops_total, elapsed

x, iters, ops, elapsed = modified_newton_method(x0)
print("\nb) Modified Newton's method")
print(f'   x = {x}')
print(f'   iterations = {iters}')
print(f'   amount of arithmetic operations = {ops}')
print(f'   elapsed time = {elapsed}')

# c) Hybrid Newton's method:
#    first k iterations: recompute Jacobian matrix and LU-decomposition each iteration
#    afterwards: freeze Jacobian matrix at last point x_(k-1) and reuse same LU factors
def hybrid_newton_method(x0, k=1, eps=1e-8, max_iter=500):
    x = x0.copy().astype(float)
    x = x0.copy().astype(float)
    ops_total = 0
    iters = 0

    t0 = time.perf_counter()

    P = L = U = None

    for i in range(max_iter):
        Fx = F(x)

        if not np.isfinite(Fx).all():
            print("Non-finite values in F(x)")
            break

        if np.linalg.norm(Fx, ord=np.inf) < eps:
            break

        # Full Newton
        if i < k:
            P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
            ops_total += ops_LU

        # else:
        # Modified Newton method;
        # freeze Jacobian matrix at current point and reuse last LU factors

        dx, ops_solve = solve_LU(P, L, U, -Fx)

        x += dx
        ops_total += ops_solve
        iters += 1

    elapsed = time.perf_counter() - t0
    return x, iters, ops_total, elapsed

def hybrid_newton_method_auto(x0, eps=1e-8, max_iter=500):
    x = x0.copy().astype(float)
    x = x0.copy().astype(float)
    ops_total = 0
    iters = 0

    t0 = time.perf_counter()

    P = L = U = None

    reuse_jacobian = False

    for i in range(max_iter):
        Fx = F(x)

        if not np.isfinite(Fx).all():
            print("Non-finite values in F(x)")
            break

        if np.linalg.norm(Fx, ord=np.inf) < eps:
            break

        # Automatic switch between full Newton and modified Newton methods
        # Jacobian changes insignificantly
        if np.linalg.norm(Fx, ord=np.inf) < 1e-4:
            reuse_jacobian = True

        # Full Newton
        if (i < k) and not reuse_jacobian:
            P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
            ops_total += ops_LU

        # else: reuse frozen LU factors

        dx, ops_solve = solve_LU(P, L, U, -Fx)

        x += dx
        ops_total += ops_solve
        iters += 1

    elapsed = time.perf_counter() - t0
    return x, iters, ops_total, elapsed

print("\nc) Hybrid Newton's method")
# k = 1 => modified Newton's method
for k in range(1, 12):
    x, iters, ops, elapsed = hybrid_newton_method(x0)
    print(f'   k = {k}, elapsed time = {elapsed}')

print("\n   Hybrid Newton's automatic method")
x, iters, ops, elapsed = hybrid_newton_method(x0)
print(f'   elapsed time = {elapsed}')

# d) Cyclic Newton method: recompute Jacobian matrix every m iterations
# m = 1 => full Newton's; m -> inf => modified Newton's; intermediate m => hybrid
def cyclic_newton_method(x0, m=1, eps=1e-8, max_iter=500):
    x = x0.copy().astype(float)
    x = x0.copy().astype(float)
    ops_total = 0
    iters = 0

    t0 = time.perf_counter()

    P = L = U = None

    for i in range(max_iter):
        Fx = F(x)

        if not np.isfinite(Fx).all():
            print("Non-finite values in F(x)")
            break

        if np.linalg.norm(Fx, ord=np.inf) < eps:
            break

        if i % m == 0:
            P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
            ops_total += ops_LU

        # else:
        # Modified Newton method;
        # freeze Jacobian matrix at current point and reuse last LU factors

        dx, ops_solve = solve_LU(P, L, U, -Fx)

        x += dx
        ops_total += ops_solve
        iters += 1

    elapsed = time.perf_counter() - t0
    return x, iters, ops_total, elapsed

print("\nd) Cyclic Newton's method")
for m in range(1, 12):
    x, iters, ops, elapsed = cyclic_newton_method(x0, m)
    print(f'   m = {m}, iterations = {iters}, operations = {ops}, time = {elapsed}')

# e) Newton's method hybrid + cyclic (use parameters k, m)
# k = inf, m = 1: full Newton; k = 1, m = inf: modified Newton
# First k iters: full Newton; afterwards: cyclic Newton every m iters
def hybrid_cyclic_newton_method(x0, k=np.inf, m=1, eps=1e-8, max_iter=500):
    x = x0.copy().astype(float)
    x = x0.copy().astype(float)
    ops_total = 0
    iters = 0

    t0 = time.perf_counter()

    P = L = U = None

    for i in range(max_iter):
        Fx = F(x)

        if not np.isfinite(Fx).all():
            print("Non-finite values in F(x)")
            break

        if np.linalg.norm(Fx, ord=np.inf) < eps:
            break

        if i < k:
            P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
            ops_total += ops_LU
        elif (i - k) % m == 0:
            P, L, U, permut, ops_LU = LU_decomposition_partial_pivot(J(x))
            ops_total += ops_LU

        # else:
        # Modified Newton method;
        # freeze Jacobian matrix at current point and reuse last LU factors

        dx, ops_solve = solve_LU(P, L, U, -Fx)

        x += dx
        ops_total += ops_solve
        iters += 1

    elapsed = time.perf_counter() - t0
    return x, iters, ops_total, elapsed

print("\ne) Hybrid + Cyclic (k, m parameters) Newton's method")
k, m = np.inf, 1
x, iters, ops, elapsed = hybrid_cyclic_newton_method(x0, k, m)
print(f'   k = {k}, m = {m}, elapsed time = {elapsed}')  # full Newton

m, k = np.inf, 1
x, iters, ops, elapsed = hybrid_cyclic_newton_method(x0, k, m)
print(f'   k = {k}, m = {m}, elapsed time = {elapsed}')  # modified Newton

# f) x_0's 5th elem: -0.5 -> -0.2
x0 = np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5], dtype=float)

print('\nf)')
x, iters, ops, elapsed = hybrid_newton_method(x0, k=1)  # k<7
print(f'   k = {1}, elapsed time = {elapsed}')
print(f'    residual norm = {np.linalg.norm(F(x), ord=np.inf)}')

x, iters, ops, elapsed = hybrid_newton_method(x0, k=7)  # k=7
print(f'   k = {7}, elapsed time = {elapsed}')
print(f'    residual norm =, {np.linalg.norm(F(x), ord=np.inf)}')

x, iters, ops, elapsed = hybrid_newton_method(x0, k=np.inf)  #k>7
print(f'   k = {np.inf}, elapsed time = {elapsed}')
print(f'    residual norm =, {np.linalg.norm(F(x), ord=np.inf)}')