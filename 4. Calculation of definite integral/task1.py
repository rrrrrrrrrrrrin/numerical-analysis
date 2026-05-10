import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================== 1 =================================================
# Приближенно вычислить интеграл f(x)dx, [a, b] (S)
# составными квадратурными формулами (математический инструмент для
# приближенного вычисления определенного интеграл):
# 1) среднего прямоугольника
# 2) трапеций
# 3) Симпсона
#
# 1. Относительная погрешность rtol = 1e-10
# Оценка погрешности по правилу Рунге
#
# 2. Начальный шаг это длина отрезка
# На каждой итерации увеличивать число шагов вдвое
#
# I — точное значение интеграла
# N — число шагов разбиения отрезка [a, b]
# h — длина шага
#
# 3. Вычислить:
#    S — приближённое значение интеграла
#    E = |I - S|/|I| — точная относительная погрешность
#    R — оценка относительной погрешности по Рунге
#    p — оценка порядка сходимости по Эйткену
#    S' = S + R — уточнение решения по Ричардсону
#    E' — точная относительная погрешность S'
#    p' — оценка порядка сходимости по Эйткуну для S'
#    pr — наклон графика относительных погрешностей R


# 1) Найдем точное значение интеграла I
a, b = 0.0, 1.0

def f(x):
    return np.exp(x)  # e^x

# e^x dx, [0, 1] = e - 1
I = np.e - 1

# 2) Составная формула среднего прямоугольника
def midpoint_rule(f, a, b, N):
    h = (b - a) / N  # длина шага (подотрезка)
    x = a + h * (np.arange(N) + 0.5)  # середины отрезков
    return h * np.sum(f(x))  # формула интегрирования

# 3) Составная формула трапеций
#    Использовать значения с предыдущей сетки
#    Формула трапеции при удвоении числа шагов:
#    T_(2N) = 1/2 * T_N + h_(2N) * sum f(new midpoints)
def trapezoidal_rule(T_prev, f, a, b, N_prev):
    h_prev = (b - a) / N_prev
    h_new = h_prev / 2
    x_new = a + h_prev * (np.arange(N_prev) + 0.5)
    T_new = 0.5 * T_prev + h_new * np.sum(f(x_new))


    return T_new

# 4) Составная формула Симпсона из формулы трапеций
def simpson_from_trapezoids(T_prev, T_new):
    # S_(2N) = (4*T_(2N) - T_N)/3
    return (4 * T_new - T_prev) / 3

# 5) Оценка порядка сходимости по Эйткену p
def aitken_method(S1, S2, S3):
    numerator = abs(S3 - S2)
    denominator = abs(S2 - S1)

    # Проверка деления на ноль
    if denominator == 0 or numerator == 0:
        return np.nan

    # Формула Эйткена
    return np.log2(denominator / numerator)

# 6) Построение таблицы для метода среднего прямоугольника
def table_midpoint(rtol=1e-10):
    rows = []
    S_vals = []
    Sr_vals = []

    N = 1
    while True:
        h = (b - a) / N
        S = midpoint_rule(f, a, b, N)
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
            S_prev = S_vals[-2]
            delta = (S - S_prev) / (2**2 - 1)   # p = 2
            R = abs(delta) / abs(S)
            S_rich = S + delta

            row["R"] = R
            row["S'"] = S_rich
            row["E'"] = abs(I - S_rich) / abs(I)
            Sr_vals.append(S_rich)

            if len(S_vals) >= 3:
                row["p"] = aitken(S_vals[-3], S_vals[-2], S_vals[-1])

            if len(Sr_vals) >= 3:
                row["p'"] = aitken(Sr_vals[-3], Sr_vals[-2], Sr_vals[-1])

            if len(rows) >= 1 and rows[-1]["R"] > 0 and R > 0:
                row["pr"] = (np.log10(R) - np.log10(rows[-1]["R"])) / (np.log10(h) - np.log10(rows[-1]["h"]))

        rows.append(row)

        if not np.isnan(row["R"]) and row["R"] < rtol:
            break

        N *= 2

    return pd.DataFrame(rows)

# 7) Построение таблицы для метода трапеций
def table_trapezoid(rtol=1e-10):
    rows = []
    S_vals = []
    Sr_vals = []

    N = 1
    h = b - a
    S = h * (f(a) + f(b)) / 2  # T_1
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
    rows.append(row)

    while True:
        N_new = 2 * N
        h_new = h / 2
        x_new = a + h * (np.arange(N) + 0.5)
        S_new = 0.5 * S + h_new * np.sum(f(x_new))   # T_{2N}

        S_vals.append(S_new)

        delta = (S_new - S) / (2**2 - 1)   # p = 2
        R = abs(delta) / abs(S_new)
        S_rich = S_new + delta

        row = {
            "N": N_new,
            "h": h_new,
            "S": S_new,
            "E": abs(I - S_new) / abs(I),
            "R": R,
            "p": np.nan,
            "S'": S_rich,
            "E'": abs(I - S_rich) / abs(I),
            "p'": np.nan,
            "pr": np.nan
        }

        Sr_vals.append(S_rich)

        if len(S_vals) >= 3:
            row["p"] = aitken(S_vals[-3], S_vals[-2], S_vals[-1])

        if len(Sr_vals) >= 3:
            row["p'"] = aitken(Sr_vals[-3], Sr_vals[-2], Sr_vals[-1])

        if len(rows) >= 1 and rows[-1]["R"] > 0 and R > 0:
            row["pr"] = (np.log10(R) - np.log10(rows[-1]["R"])) / (np.log10(h_new) - np.log10(rows[-1]["h"]))

        rows.append(row)

        if R < rtol:
            break

        N, h, S = N_new, h_new, S_new

    return pd.DataFrame(rows)


# 8) Построение таблицы для метода Симпсона
def table_simpson(rtol=1e-10):
    rows = []
    S_vals = []
    Sr_vals = []

    # Начинаем с N=2, потому что для Симпсона число шагов должно быть чётным
    N = 2
    h = (b - a) / N

    # T_1
    T_prev = (b - a) * (f(a) + f(b)) / 2

    # T_2
    x = np.linspace(a, b, N + 1)
    y = f(x)
    T = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    # S_2
    S = simpson_from_trapezoids(T_prev, T)
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
    rows.append(row)

    while True:
        N_new = 2 * N
        h_new = h / 2

        x_new = a + h * (np.arange(N) + 0.5)
        T_new = 0.5 * T + h_new * np.sum(f(x_new))  # T_{2N}
        S_new = simpson_from_trapezoids(T, T_new)    # S_{2N}

        S_vals.append(S_new)

        delta = (S_new - S) / (2**4 - 1)   # p = 4
        R = abs(delta) / abs(S_new)
        S_rich = S_new + delta

        row = {
            "N": N_new,
            "h": h_new,
            "S": S_new,
            "E": abs(I - S_new) / abs(I),
            "R": R,
            "p": np.nan,
            "S'": S_rich,
            "E'": abs(I - S_rich) / abs(I),
            "p'": np.nan,
            "pr": np.nan
        }
        Sr_vals.append(S_rich)

        if len(S_vals) >= 3:
            row["p"] = aitken(S_vals[-3], S_vals[-2], S_vals[-1])

        if len(Sr_vals) >= 3:
            row["p'"] = aitken(Sr_vals[-3], Sr_vals[-2], Sr_vals[-1])

        if len(rows) >= 1 and rows[-1]["R"] > 0 and R > 0:
            row["pr"] = (np.log10(R) - np.log10(rows[-1]["R"])) / (np.log10(h_new) - np.log10(rows[-1]["h"]))

        rows.append(row)

        if R < rtol:
            break

        N, h, T, S = N_new, h_new, T_new, S_new

    return pd.DataFrame(rows)


# 9) Печать таблицы
def print_table(df, title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    out = df.copy()
    out = out.replace({np.nan: ""})  # Nan -> пустая строка
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


# График R(h) в двойной логарифмической шкале
# (зависимость log_10(R) от log_10(h)) для каждой формулы
#
#
def plot_R(df, title, pbar):
    d = df.dropna(subset=["R", "h"]).copy()
    h = d["h"].to_numpy()
    R = d["R"].to_numpy()

    x = np.log10(h)
    y = np.log10(R)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-", label="log10(R)")

    # Опорная прямая с наклоном pbar
    c = np.median(R / (h ** pbar))
    h_line = np.array([h.min(), h.max()])
    plt.plot(np.log10(h_line), np.log10(c * h_line**pbar), "k--", label=f"опорная прямая, наклон {pbar}")

    plt.xlabel("log10(h)")
    plt.ylabel("log10(R)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


mid_df = table_midpoint()
trap_df = table_trapezoid()
simp_df = table_simpson()

print_table(mid_df, "Средний прямоугольник")
print_table(trap_df, "Трапеции")
print_table(simp_df, "Симпсон")

plot_R(mid_df, "Средний прямоугольник: log10(R) от log10(h)", 2)
plot_R(trap_df, "Трапеции: log10(R) от log10(h)", 2)
plot_R(simp_df, "Симпсон: log10(R) от log10(h)", 4)

# ========================================== 2 ============================================
def find_nopt(method, pbar, df, rtol, N_min, N_max, even=False):
    # 1) берём первую строку, где p уже близко к теоретическому
    good = df[(df["p"].notna()) & (abs(df["p"] - pbar) < 0.05 * pbar)]
    row = good.iloc[0]

    # 2) оценка Nopt по Рунге
    N_est = int(np.ceil(row["N"] * (row["R"] / rtol) ** (1 / pbar)))

    if even and N_est % 2 != 0:
        N_est += 1

    # 3) проверка и при необходимости увеличиваем N
    N = max(N_est, N_min)
    if even and N % 2 != 0:
        N += 1

    while N <= N_max:
        S = method(f, a, b, N)
        E = abs(I - S) / abs(I)
        if E < rtol:
            return row["N"], N_est, N, E
        N += 2 if even else 1

    return row["N"], N_est, None, None


# Средние прямоугольники
N_used_mid, N_est_mid, Nopt_mid, E_mid = find_nopt(
    midpoint_rule, 2, mid_df, rtol, 16384, 32768, even=False
)

# Трапеции
N_used_tr, N_est_tr, Nopt_tr, E_tr = find_nopt(
    trapezoidal_rule, 2, trap_df, rtol, 16384, 32768, even=False
)

# Симпсон
N_used_sim, N_est_sim, Nopt_sim, E_sim = find_nopt(
    simpson_rule, 4, simp_df, rtol, 64, 128, even=True
)

print("Midpoint:", N_used_mid, N_est_mid, Nopt_mid, E_mid)
print("Trapezoid:", N_used_tr, N_est_tr, Nopt_tr, E_tr)
print("Simpson:", N_used_sim, N_est_sim, Nopt_sim, E_sim)

