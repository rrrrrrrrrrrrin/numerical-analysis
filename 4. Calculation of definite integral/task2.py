import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


a, b = 0.0, 1.0

def f(x):
    return np.exp(x)

def weight(x):
    return np.ones_like(x, dtype=float)

I = np.e - 1
rtol = 1e-10


# ============================================================
# 2) ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ ДЛЯ МОМЕНТОВ ВЕСА
# ============================================================
def simpson_integral(func, A, B, n=400):
    """Составной Симпсон для численного интегрирования."""
    if n % 2 == 1:
        n += 1
    x = np.linspace(A, B, n + 1)
    y = func(x)
    h = (B - A) / n
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


def moment_weighted(k, A, B):
    """μ_k = ∫_A^B x^k p(x) dx"""
    return simpson_integral(lambda x: (x**k) * weight(x), A, B, n=400)


# ============================================================
# 3) 3-точечная формула Ньютона–Котса на одном отрезке [A,B]
#    Узлы: A, (A+B)/2, B
#    Коэффициенты зависят от веса p(x)
# ============================================================
def nc3_local(A, B):
    h = B - A
    x0 = A
    x1 = (A + B) / 2
    x2 = B

    # Базис Лагранжа на t in [0,1], x = A + h t
    # l0(t) = 2t^2 - 3t + 1
    # l1(t) = -4t^2 + 4t
    # l2(t) = 2t^2 - t
    #
    # Весовые коэффициенты:
    # α_i = ∫_A^B p(x) l_i((x-A)/h) dx
    #
    # Перепишем l_i через x:
    # t = (x-A)/h
    def l0(x):
        t = (x - A) / h
        return 2*t*t - 3*t + 1

    def l1(x):
        t = (x - A) / h
        return -4*t*t + 4*t

    def l2(x):
        t = (x - A) / h
        return 2*t*t - t

    a0 = simpson_integral(lambda x: weight(x) * l0(x), A, B, n=300)
    a1 = simpson_integral(lambda x: weight(x) * l1(x), A, B, n=300)
    a2 = simpson_integral(lambda x: weight(x) * l2(x), A, B, n=300)

    return a0 * f(x0) + a1 * f(x1) + a2 * f(x2)


# ============================================================
# 4) 3-точечная формула Гаусса на одном отрезке [A,B]
#    Строится по моментам весовой функции на [A,B]
# ============================================================
def gauss3_local(A, B):
    # Моменты μ_0 ... μ_6
    mu = np.array([moment_weighted(k, A, B) for k in range(7)], dtype=float)

    # Ищем многочлен q(x)=x^3+c2 x^2+c1 x+c0,
    # ортогональный 1, x, x^2 относительно веса p(x):
    # ∫ q(x) x^m p(x) dx = 0, m=0,1,2
    # => система по c0,c1,c2:
    # μ_{m+3} + c2 μ_{m+2} + c1 μ_{m+1} + c0 μ_m = 0
    M = np.array([
        [mu[2], mu[1], mu[0]],
        [mu[3], mu[2], mu[1]],
        [mu[4], mu[3], mu[2]],
    ], dtype=float)
    rhs = -np.array([mu[3], mu[4], mu[5]], dtype=float)

    c2, c1, c0 = np.linalg.solve(M, rhs)

    # Корни ортогонального многочлена — узлы Гаусса
    roots = np.roots([1.0, c2, c1, c0])
    roots = np.sort(np.real_if_close(roots).astype(float))

    # Весовые коэффициенты из системы точности на 1, x, x^2
    V = np.vstack([np.ones(3), roots, roots**2]).T
    w = np.linalg.solve(V, mu[:3])

    return np.sum(w * f(roots))


# ============================================================
# 5) Составные формулы
# ============================================================
def composite_nc3(N):
    """Составная 3-точечная Ньютона–Котса."""
    x = np.linspace(a, b, N + 1)
    S = 0.0
    for i in range(N):
        S += nc3_local(x[i], x[i + 1])
    return S


def composite_gauss3(N):
    """Составная 3-точечная Гаусса."""
    x = np.linspace(a, b, N + 1)
    S = 0.0
    for i in range(N):
        S += gauss3_local(x[i], x[i + 1])
    return S


# ============================================================
# 6) Оценка p по Эйткену и таблица
# ============================================================
def aitken_p(S1, S2, S3):
    num = abs(S3 - S2)
    den = abs(S2 - S1)
    if num == 0 or den == 0:
        return np.nan
    return np.log2(den / num)


def build_table(method, method_name, rtol, p_ref_low=None, p_ref_high=None):
    rows = []
    S_vals = []
    Sr_vals = []

    N = 1
    while True:
        h = (b - a) / N
        S = method(N)
        S_vals.append(S)

        row = {
            "N": N,
            "h": h,
            "S": S,
            "E": abs(I - S) / abs(I),
            "R": np.nan,
            "p": np.nan,
            "S'": np.nan,
            "E'": np.nan,
            "p'": np.nan,
            "pr": np.nan
        }

        if len(S_vals) >= 2:
            # Для оценки Рунге используем p из предыдущей аппроксимации,
            # если оно уже стабилизировалось.
            if len(S_vals) >= 3:
                p_est = aitken_p(S_vals[-3], S_vals[-2], S_vals[-1])
                row["p"] = p_est
            else:
                p_est = None

            # Если p ещё нет, берём теоретический ориентир
            if p_est is None or np.isnan(p_est):
                if method_name == "NC3":
                    p_used = 3.5
                else:
                    p_used = 6.0
            else:
                p_used = p_est

            S_prev = S_vals[-2]
            delta = (S - S_prev) / (2**p_used - 1)
            R = abs(delta) / abs(S)
            S_rich = S + delta

            row["R"] = R
            row["S'"] = S_rich
            row["E'"] = abs(I - S_rich) / abs(I)

            Sr_vals.append(S_rich)

            if len(Sr_vals) >= 3:
                row["p'"] = aitken_p(Sr_vals[-3], Sr_vals[-2], Sr_vals[-1])

            if len(rows) >= 1 and rows[-1]["R"] > 0 and R > 0:
                row["pr"] = (np.log10(R) - np.log10(rows[-1]["R"])) / (
                    np.log10(h) - np.log10(rows[-1]["h"])
                )

        rows.append(row)

        # Условие остановки по Рунге
        if not np.isnan(row["R"]) and row["R"] < rtol:
            break

        N *= 2

    df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print(method_name)
    print("=" * 90)

    out = df.copy().replace({np.nan: ""})
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.12e}".format)
    print(out.to_string(index=False))

    return df


# ============================================================
# 7) График log10(R) от log10(h)
# ============================================================
def plot_R(df, title, slope1, slope2=None):
    d = df.dropna(subset=["R", "h"]).copy()
    h = d["h"].to_numpy()
    R = d["R"].to_numpy()

    x = np.log10(h)
    y = np.log10(R)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-", label="log10(R)")

    # опорная прямая со slope1
    c1 = np.median(R / (h ** slope1))
    h_line = np.array([h.min(), h.max()])
    plt.plot(np.log10(h_line), np.log10(c1 * h_line**slope1),
             "k--", label=f"опорная прямая, наклон {slope1}")

    if slope2 is not None:
        c2 = np.median(R / (h ** slope2))
        plt.plot(np.log10(h_line), np.log10(c2 * h_line**slope2),
                 "k--", alpha=0.6, label=f"опорная прямая, наклон {slope2}")

    plt.xlabel("log10(h)")
    plt.ylabel("log10(R)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ============================================================
# 8) Подбор Nopt и проверка точности
# ============================================================
def find_nopt(method, df, p_min, p_max, is_gauss=False):
    # Берём первую строку, где p попал в требуемый диапазон
    good = df[(df["p"].notna()) & (df["p"] > p_min) & (df["p"] < p_max)].copy()
    if good.empty:
        return None

    row = good.iloc[0]
    N0 = int(row["N"])
    R0 = float(row["R"])
    p0 = float(row["p"])

    # оценка Nopt по Рунге
    N_est = int(np.ceil(N0 * (R0 / rtol) ** (1.0 / p0)))

    # для симметричных формул N должен быть кратен 2, но здесь N — число отрезков,
    # а в нашем построении каждый отрезок считается отдельно, так что ограничение не нужно

    # увеличиваем, пока точная погрешность не станет достаточной
    N = max(N_est, N0 + 1)
    while True:
        S = method(N)
        E = abs(I - S) / abs(I)
        if E < rtol:
            return {
                "Nrow": N0,
                "p": p0,
                "N_est": N_est,
                "Nopt": N,
                "E": E
            }
        N += 1


# ============================================================
# 9) Запуск
# ============================================================
nc_df = build_table(composite_nc3, "NC3", rtol)
gauss_df = build_table(composite_gauss3, "Gauss3", rtol)

# Графики
plot_R(nc_df, "Newton-Cotes 3-point: log10(R) vs log10(h)", 3, 4)
plot_R(gauss_df, "Gauss 3-point: log10(R) vs log10(h)", 6)

# Подбор Nopt
nc_opt = find_nopt(composite_nc3, nc_df, 2.7, 4.0)
gauss_opt = find_nopt(composite_gauss3, gauss_df, 5.7, 6.7, is_gauss=True)

print("\nNopt results")
print("NC3   :", nc_opt)
print("Gauss :", gauss_opt)