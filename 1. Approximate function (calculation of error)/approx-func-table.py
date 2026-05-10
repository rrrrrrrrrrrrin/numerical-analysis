import numpy as np
import math

# 1. Taylor series for cos and arctan
def cos_taylor(x, e=10**-6/2.88, stop=1000):
    s = np.float64(1.0)
    term = np.float64(1.0)
    for m in range(1, stop):
        term *= -x*x / ((2.0*m - 1.0) * 2.0*m)
        s += term
        if abs(s) < e: break
    return s

# |x| >= 1
def arctan_taylor(x, e=10**-6/1.44, stop=2000):
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x
    if x < 1.0:
        s = np.float64(x)
        term = np.float64(x)
        for m in range(1, stop):
            term *= -x * x
            add = term / (2.0 * m + 1.0)
            s += add
            if abs(add) < e: break
        return sign * s
    else:
        return sign * (math.pi / 2 - arctan_taylor(1.0 / x))

# 2. Heron's formula for sqrt
def sq(x, e=10**-6/2.25, stop=300):
    s = np.float64(0)
    w = np.float64(0.36)
    for m in range(stop):
        w_new = np.float64(0.5*(w + x/w))
        if abs(w_new - w) < e: return w_new
        w = w_new
    return w


# 3. Output
print("x, z_precise, z_approximate, |z_error|")

for i in range(11):
    x = round(np.float64(0.5 + 0.01*i), 2)

    # precise
    v = math.sqrt(x)
    u = math.cos(0.5 + v)
    w = math.atan(1 + 2*x*v)
    z = u/w

    # approximate
    v_approx = sq(x)
    u_approx = cos_taylor(0.5 + v_approx)
    w_approx = arctan_taylor(1 + 2*x*v_approx)
    z_approx = u_approx/w_approx

    # error
    z_err = abs(z - z_approx)

    # print
    print(x, z, z_approx, z_err)
