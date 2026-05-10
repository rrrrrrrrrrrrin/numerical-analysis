import numpy as np
import random as rnd


# ========================================== 4. ===========================================
# Decompose A matrix: A = D + L + U
# D - diag matrix (main diag elems of A)
# U, L - upper- and lower- triangular matrices (elems of A); their main diag is filled with zeros
#
# b = (b1, ..., bn)^T
#
# Jacobi method (x = x_i; element-wise): x(k+1) = D_inv * (b - (L + U)x(k)
#
# Gauss-Seidel method: x(k+1) = (D + L)_inv * (b - Ux(k))

def Jacobi_for_xi(A, b, x):
    n = len(A)
    # x_res is an array of zeros with the same shape and data type as x
    x_new = np.zeros_like(x)
    for i in range(n):
        s = b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]
        x_new[i] = s / A[i, i]
    return x_new

def Seidel_for_xi(A, b, x):
    n = len(A)
    x_new = x.copy()
    for i in range(n):
        s1 = np.dot(A[i, 0:i], x_new[0:i])
        s2 = np.dot(A[i, i+1:n], x[i+1:])
        x_new[i] = (b[i] - s1 - s2) / A[i,i]
    return x_new

# To find ||B|| = q
def Jacobi_iter_matrix(A):
    # Iteration matrix is B = -D_inv * (L + U)
    D = np.diag(np.diag(A))  # 2D matrix
    R = A - D  # R = A - D = L + U

    # np.linalg.solve does not compute D_inv explicitly
    # it solves a linear system DB = -R column by column
    B = -np.linalg.solve(D, R)
    return B

# To find ||B|| = q
def Seidel_iter_matrix(A):
    # Iteration matrix is B = -(D + L)_inv * U
    D = np.diag(np.diag(A))  # 2D matrix
    L = np.tril(A, -1)  # lower-triangular matrix; elems below diag (-1)
    U = np.triu(A, 1)  # upper-triangular matrix; elems above dig (1)
    B = -np.linalg.solve(D + L, U)
    return B

# Compare a priori and a posteriori estimation of the iteration count
# that was sufficient by the stopping rule
#
# Contraction factor: 0 < ||B|| = q < 1 (знаменатель геом. прогрессии)
#
# x1_x0_norm = ||x(1) - x(0)||
# A priori estimation: ||x(k) - x(*)|| <= q^k * x1_x0_norm/(1-q)
# Stopping rule: q^k/(1-q) * x1_x0_norm <= eps => isolate k via log
# x0 and x1 represent initial guess and first iterate
# A priori estimation predicts in advance how many iterations will be needed
def priori_estimation(q, x1_x0_norm, eps=1e-12):
    if not (0 < q < 1):
        return None  # estimation is not valid

    k = np.log(eps * (1-q) / x1_x0_norm) / np.log(q)  # k >= stopping rule
    return int(np.ceil(k))  # need the smallest integer

# diff_norm = ||x(k+1) - x(k)||
# A posteriori estimation: ||x(k) - x(*)|| <= q * diff_norm/(1-q)
# Stopping rule: q/(1-q) * diff_norm <= eps
# diff_norm (k indices) => calculate diff_norm in solve_iterative below
# A posteriori estimate is based on the current difference between two successive iterates
def posteriori_error_estimation(q, diff_norm):
    if not (0 < q < 1):
        return None
    return (q/(1-q)) * diff_norm

# Computes approximation x(k) to real solution x(*) by iteration
# x = Bx + c; Iteration: x(k+1) = Bx(k) + c
def solve_iterative(A, b, for_xi, q, eps=1e-12, max_iter=10000):
    n = len(b)
    x = np.zeros(n)

    for k in range(1, max_iter + 1):
        x_new = for_xi(A, b, x)  # either Jacobi or Seidel
        diff_norm = np.linalg.norm(x_new - x, ord=np.inf)

        # Stopping rule: ||x(k+1) - x(k)|| <= (1-q)/q * eps
        if diff_norm <= (1 - q) / q * eps:
            err = posteriori_error_estimation(q, diff_norm)
            return x_new, k, diff_norm, err

        x = x_new

    return x, max_iter, None, None

def generate_diag_dominant(n):
    A = np.random.uniform(-10, 10,  (n, n))

    # Make diag elems significantly bigger than the sums of other elems in row
    # A[i, i] = sum(abs(A[i, j])) + random_value
    diag = np.sum(np.abs(A), axis=1) + np.random.uniform(1, 20, n)
    np.fill_diagonal(A, diag)
    return A

def generate_pos_definite(n):
    B = np.random.uniform(-10, 10, (n, n))

    # A = B * B.T guarantees a positive-definite matrix A
    A = np.dot(B, B.T)
    return A


print('4.\n\n')
print('For a dominant matrix: \n')

n = rnd.randint(1, 10)
b = np.random.randn(n)
A_dom = generate_diag_dominant(n)
A = A_dom

# Jacobi method; used for diag dominant matrices
jacobi_B = Jacobi_iter_matrix(A)
jacobi_q = np.linalg.norm(jacobi_B, ord=np.inf)
print(f'Jacobi q = {jacobi_q}\n')

x0 = np.zeros(len(b))  # starting point for iteration
x1 = Jacobi_for_xi(A, b, x0)

x1_x0_norm = np.linalg.norm(x1 - x0, ord=np.inf)

print(f'For Jacobi method:\n')
k_apriori = priori_estimation(jacobi_q, x1_x0_norm)
print(f'A priori estimation: {k_apriori}')

x, k, diff_norm, err = solve_iterative(A, b, Jacobi_for_xi, jacobi_q)
print(f'Amount of iteration: {k}\n'
      f'Difference norm ||x(k+1)-x(k)||: {diff_norm}\n'
      f'A posteriori error estimation: {err}\n'
      f'Solution x: {x}\n')# x = Bx + c; Итерация: x(k+1) = Bx(k) + c

# Solution is x(k) (equals approximately) x(*);
# x(k) is current approx, x(*) is real solution
print(f'Approx solution Jacobi: {x}\n')
print(f'Ax = b?: {np.allclose(A @ x, b)}\n')

# Seidel method
seidel_B = Seidel_iter_matrix(A)
seidel_q = np.linalg.norm(seidel_B, ord=np.inf)
print(f'Seidel q = {seidel_q}\n')

x0 = np.zeros(len(b))
x1 = Seidel_for_xi(A, b, x0)

x1_x0_norm = np.linalg.norm(x1 - x0, ord=np.inf)

print(f'For Seidel method:\n')
k_apriori = priori_estimation(seidel_q, x1_x0_norm)
print(f'A priori estimation: {k_apriori}\n')

x, k, diff_norm, err = solve_iterative(A, b, Seidel_for_xi, seidel_q)
print(f'Amount of iteration: {k}\n'
      f'Difference norm ||x(k+1)-x(k)||: {diff_norm}\n'
      f'A posteriori error estimation: {err}\n'
      f'Solution x: {x}\n')

print(f'Approx solution Seidel: {x}\n')
print(f'Ax = b?: {np.allclose(A @ x, b)}\n\n')


print(f'For a positive-definite matrix: \n')
A_pos_def = generate_pos_definite(n)
A = A_pos_def

# Jacobi method
jacobi_B = Jacobi_iter_matrix(A)
jacobi_q = np.linalg.norm(jacobi_B, ord=np.inf)
print(f'Jacobi q = {jacobi_q}\n')

# Only call solver if matrix is convergent (сходящаяся), i.e. 0 < q < 1
if (jacobi_q >= 1):
    print('Matrix is not convergent. Skip Jacobi iteration\n')
else:
    x0 = np.zeros(len(b))  # starting point for iteration
    x1 = Jacobi_for_xi(A, b, x0)

    x1_x0_norm = np.linalg.norm(x1 - x0, ord=np.inf)

    print(f'For Jacobi method:\n')
    k_apriori = priori_estimation(jacobi_q, x1_x0_norm)
    print(f'A priori estimation: {k_apriori}')

    x, k, diff_norm, err = solve_iterative(A, b, Jacobi_for_xi, jacobi_q)
    print(f'Amount of iteration: {k}\n'
          f'Difference norm ||x(k+1)-x(k)||: {diff_norm}\n'
          f'A posteriori error estimation: {err}\n'
          f'Solution x: {x}\n')# x = Bx + c; Итерация: x(k+1) = Bx(k) + c

    # Solution is x(k) (equals approximately) x(*);
    # x(k) is current approx, x(*) is real solution
    print(f'Approx solution Jacobi: {x}\n')
    print(f'Ax = b?: {np.allclose(A @ x, b)}\n')

# Seidel method
seidel_B = Seidel_iter_matrix(A)
seidel_q = np.linalg.norm(seidel_B, ord=np.inf)
print(f'Seidel q = {seidel_q}\n')

x0 = np.zeros(len(b))
x1 = Seidel_for_xi(A, b, x0)

x1_x0_norm = np.linalg.norm(x1 - x0, ord=np.inf)

print(f'For Seidel method:\n')
k_apriori = priori_estimation(seidel_q, x1_x0_norm)
print(f'A priori estimation: {k_apriori}\n')

x, k, diff_norm, err = solve_iterative(A, b, Seidel_for_xi, seidel_q)
print(f'Amount of iteration: {k}\n'
      f'Difference norm ||x(k+1)-x(k)||: {diff_norm})\n'
      f'A posteriori error estimation: {err}\n'
      f'Solution x: {x}\n')

print(f'Approx solution Seidel: {x}\n')
print(f'Ax = b?: {np.allclose(A @ x, b)}\n\n')